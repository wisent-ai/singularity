//! Anonymous owner-only credential files cross exec, not a resident wrapper.
//! The running product keeps their descriptors close-on-exec. Only an explicit
//! Singularity child handoff may inherit them; ordinary tools receive none.

use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::fs::{FileExt, MetadataExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::sync::OnceLock;

use secrecy::SecretString;
use zeroize::Zeroizing;

use crate::AppError;

/// The variables that name the inherited descriptors, in the order the
/// credentials are read: Brama's HMAC key, Brama's bearer, Most's token.
const DESCRIPTORS: &[&str] = &[
    "SINGULARITY_BRAMA_HMAC_FD",
    "SINGULARITY_BRAMA_BEARER_FD",
    "SINGULARITY_MOST_TOKEN_FD",
];

/// What fcntl answers when it refuses; errno then says why.
const FCNTL_REFUSED: libc::c_int = -1; // https://pubs.opengroup.org/onlinepubs/9799919799/functions/fcntl.html "Otherwise, -1 shall be returned and errno set"

pub(crate) struct InheritedCredentials {
    files: [File; DESCRIPTORS.len()],
    pub(crate) brama: SecretString,
    pub(crate) bearer: SecretString,
    pub(crate) most: SecretString,
}

static INHERITED: OnceLock<InheritedCredentials> = OnceLock::new();

/// The descriptor one variable names: set, UTF-8, a number, and not one of
/// the standard streams.
fn named_descriptor(name: &str, value: Option<&std::ffi::OsStr>) -> Result<RawFd, AppError> {
    let written = match value.map(|value| value.to_str()) {
        Some(Some(written)) => written,
        Some(None) => return Err(AppError::Secret(format!("{name} is not valid UTF-8"))),
        None => {
            return Err(AppError::Secret(format!(
                "{name} is unset while the other bootstrap descriptors are set"
            )));
        }
    };
    let descriptor = written.parse::<RawFd>().map_err(|error| {
        AppError::Secret(format!(
            "{name} {written:?} is not a descriptor number: {error}"
        ))
    })?;
    if descriptor <= libc::STDERR_FILENO {
        return Err(AppError::Secret(format!(
            "{name} names descriptor {descriptor}, a standard stream rather than an inherited credential"
        )));
    }
    Ok(descriptor)
}

/// The credential one inherited file holds, read whole from its first byte.
fn read_credential(name: &str, file: &File) -> Result<SecretString, AppError> {
    let observed_length = file
        .metadata()
        .map_err(|error| {
            AppError::Secret(format!(
                "{name}: inspecting inherited credential failed: {error}"
            ))
        })?
        .len();
    let length = usize::try_from(observed_length).map_err(|error| {
        AppError::Secret(format!(
            "{name}: credential length {observed_length} cannot be represented: {error}"
        ))
    })?;
    let mut bytes = Zeroizing::new(Vec::new());
    bytes.try_reserve_exact(length).map_err(|error| {
        AppError::Secret(format!(
            "{name}: reserving {length} credential bytes failed: {error}"
        ))
    })?;
    bytes.resize_with(length, Default::default);
    let start = 0; // https://pubs.opengroup.org/onlinepubs/9799919799/functions/pread.html "read from a given position in the file": the file's first byte
    file.read_exact_at(&mut bytes, start).map_err(|error| {
        AppError::Secret(format!(
            "{name}: reading {length} inherited credential bytes failed: {error}"
        ))
    })?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| AppError::Secret(format!("{name} is not UTF-8: {error}")))?;
    crate::config::secret_value(text, name)
}

/// Adopt the bootstrap handoff before starting threads or tool processes.
/// The files were unlinked before exec, so a signal or crash cannot leave
/// their secrets behind in a runtime directory.
pub fn adopt_credentials() -> Result<(), AppError> {
    let values: Vec<_> = DESCRIPTORS
        .iter()
        .map(|name| std::env::var_os(name))
        .collect();
    if values.iter().all(Option::is_none) {
        return Ok(());
    }
    if INHERITED.get().is_some() {
        return Err(AppError::Secret(
            "bootstrap credentials were already adopted".into(),
        ));
    }
    let mut descriptors = [RawFd::default(); DESCRIPTORS.len()];
    for (index, (name, value)) in DESCRIPTORS.iter().zip(&values).enumerate() {
        let descriptor = named_descriptor(name, value.as_deref())?;
        if descriptors[..index].contains(&descriptor) {
            return Err(AppError::Secret(
                "bootstrap credential descriptors must be distinct".into(),
            ));
        }
        // SAFETY: fcntl validates the borrowed descriptor without dereferencing memory.
        let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
        if flags == FCNTL_REFUSED {
            return Err(AppError::Secret(format!(
                "{name}: {}",
                std::io::Error::last_os_error()
            )));
        }
        descriptors[index] = descriptor;
    }
    // SAFETY: all descriptors were checked above, are distinct, and have no
    // Rust owner in this new process image. Adoption runs before other threads.
    let files = descriptors.map(|descriptor| unsafe { File::from_raw_fd(descriptor) });
    for (name, file) in DESCRIPTORS.iter().zip(&files) {
        let metadata = file.metadata().map_err(|error| {
            AppError::Secret(format!(
                "{name}: inspecting inherited descriptor {} failed: {error}",
                file.as_raw_fd()
            ))
        })?;
        // SAFETY: geteuid has no arguments or memory preconditions.
        let owner = unsafe { libc::geteuid() };
        // An anonymous credential file is named by no directory: no links.
        let linked = metadata.nlink() != 0;
        let mode = metadata.permissions().mode();
        let shared = mode & crate::config::GROUP_OR_OTHER_ACCESS != 0;
        let empty = metadata.len() == 0;
        if !metadata.is_file() || linked || metadata.uid() != owner || shared || empty {
            return Err(AppError::Secret(format!(
                "{name}: inherited descriptor {} must be an anonymous, nonempty, owner-only regular file; observed regular={}, links={}, uid={}, effective_uid={owner}, mode={:o}, bytes={}",
                file.as_raw_fd(),
                metadata.is_file(),
                metadata.nlink(),
                metadata.uid(),
                metadata.permissions().mode(),
                metadata.len()
            )));
        }
        close_on_exec(file.as_raw_fd(), true).map_err(|error| {
            AppError::Secret(format!(
                "{name}: protecting inherited descriptor {} from child inheritance failed: {error}",
                file.as_raw_fd()
            ))
        })?;
    }
    let secrets = DESCRIPTORS
        .iter()
        .zip(&files)
        .map(|(name, file)| read_credential(name, file))
        .collect::<Result<Vec<_>, _>>()?;
    let [brama, bearer, most]: [SecretString; DESCRIPTORS.len()] =
        secrets.try_into().map_err(|read: Vec<SecretString>| {
            AppError::Secret(format!(
                "{} bootstrap credentials were read for {} descriptors",
                read.len(),
                DESCRIPTORS.len()
            ))
        })?;
    INHERITED
        .set(InheritedCredentials {
            files,
            brama,
            bearer,
            most,
        })
        .map_err(|_| AppError::Secret("bootstrap credentials were already adopted".into()))
}

pub(crate) fn inherited_credentials() -> Result<Option<&'static InheritedCredentials>, AppError> {
    if let Some(credentials) = INHERITED.get() {
        return Ok(Some(credentials));
    }
    if DESCRIPTORS
        .iter()
        .any(|name| std::env::var_os(name).is_some())
    {
        return Err(AppError::Secret(
            "bootstrap credential handoff was not adopted before runtime startup".into(),
        ));
    }
    Ok(None)
}

pub(crate) fn inherit_for_child(command: &mut Command) -> Result<(), AppError> {
    if let Some(credentials) = inherited_credentials()? {
        pass_files(command, &credentials.files);
    }
    Ok(())
}

pub(super) fn pass_files(command: &mut Command, files: &[File; DESCRIPTORS.len()]) {
    let descriptors = files.each_ref().map(AsRawFd::as_raw_fd);
    for (name, descriptor) in DESCRIPTORS.iter().zip(descriptors) {
        command.env(name, descriptor.to_string());
    }
    // SAFETY: the post-fork closure only performs fcntl calls on owned file
    // descriptors. No allocation, locking, environment access or Rust cleanup.
    unsafe {
        command.pre_exec(move || {
            for descriptor in descriptors {
                close_on_exec(descriptor, false)?;
            }
            Ok(())
        });
    }
}

fn close_on_exec(descriptor: RawFd, enabled: bool) -> std::io::Result<()> {
    // SAFETY: these fcntl operations access no Rust memory.
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
    if flags == FCNTL_REFUSED {
        return Err(std::io::Error::last_os_error());
    }
    let flags = if enabled {
        flags | libc::FD_CLOEXEC
    } else {
        flags & !libc::FD_CLOEXEC
    };
    // SAFETY: the descriptor and flag word were checked above.
    if unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags) } == FCNTL_REFUSED {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
