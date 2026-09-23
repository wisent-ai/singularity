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
use super::MAX_SECRET_BYTES;

const DESCRIPTORS: [&str; 3] = [
    "SINGULARITY_BRAMA_HMAC_FD",
    "SINGULARITY_BRAMA_BEARER_FD",
    "SINGULARITY_MOST_TOKEN_FD",
];

pub(crate) struct InheritedCredentials {
    files: [File; DESCRIPTORS.len()],
    pub(crate) brama: SecretString,
    pub(crate) bearer: SecretString,
    pub(crate) most: SecretString,
}

static INHERITED: OnceLock<InheritedCredentials> = OnceLock::new();

/// Adopt the bootstrap handoff before starting threads or tool processes.
/// The files were unlinked before exec, so a signal or crash cannot leave
/// their secrets behind in a runtime directory.
pub fn adopt_credentials() -> Result<(), AppError> {
    let values = DESCRIPTORS.map(std::env::var_os);
    if values.iter().all(Option::is_none) {
        return Ok(());
    }
    if INHERITED.get().is_some() {
        return Err(AppError::Secret("bootstrap credentials were already adopted".into()));
    }
    let mut descriptors = [RawFd::default(); DESCRIPTORS.len()];
    for (index, (name, value)) in DESCRIPTORS.iter().zip(values).enumerate() {
        let descriptor = value.as_deref().and_then(|value| value.to_str())
            .and_then(|value| value.parse::<RawFd>().ok())
            .filter(|descriptor| *descriptor > libc::STDERR_FILENO)
            .ok_or_else(|| AppError::Secret(format!("{name} must name an inherited credential descriptor")))?;
        if descriptors[..index].contains(&descriptor) {
            return Err(AppError::Secret("bootstrap credential descriptors must be distinct".into()));
        }
        // SAFETY: fcntl validates the borrowed descriptor without dereferencing memory.
        if unsafe { libc::fcntl(descriptor, libc::F_GETFD) } < 0 {
            return Err(AppError::Secret(format!("{name}: {}", std::io::Error::last_os_error())));
        }
        descriptors[index] = descriptor;
    }
    // SAFETY: all descriptors were checked above, are distinct, and have no
    // Rust owner in this new process image. Adoption runs before other threads.
    let files = descriptors.map(|descriptor| unsafe { File::from_raw_fd(descriptor) });
    for (name, file) in DESCRIPTORS.iter().zip(&files) {
        let metadata = file.metadata()?;
        // SAFETY: geteuid has no arguments or memory preconditions.
        let owner = unsafe { libc::geteuid() };
        if !metadata.is_file() || metadata.nlink() != 0 || metadata.uid() != owner
            || metadata.permissions().mode() & crate::config::GROUP_OR_OTHER_ACCESS != 0
            || metadata.len() == 0 || metadata.len() > MAX_SECRET_BYTES as u64
        {
            return Err(AppError::Secret(format!("{name} is not an anonymous owner-only credential file")));
        }
        close_on_exec(file.as_raw_fd(), true)?;
    }
    let credential = |index: usize| -> Result<SecretString, AppError> {
        let mut bytes = Zeroizing::new(vec![0u8; files[index].metadata()?.len() as usize]);
        files[index].read_exact_at(&mut bytes, 0)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|error| AppError::Secret(format!("{} is not UTF-8: {error}", DESCRIPTORS[index])))?;
        crate::config::secret_value(text, DESCRIPTORS[index])
    };
    let brama = credential(0)?;
    let bearer = credential(1)?;
    let most = credential(2)?;
    INHERITED.set(InheritedCredentials { files, brama, bearer, most })
        .map_err(|_| AppError::Secret("bootstrap credentials were already adopted".into()))
}

pub(crate) fn inherited_credentials() -> Result<Option<&'static InheritedCredentials>, AppError> {
    if let Some(credentials) = INHERITED.get() {
        return Ok(Some(credentials));
    }
    if DESCRIPTORS.iter().any(|name| std::env::var_os(name).is_some()) {
        return Err(AppError::Secret("bootstrap credential handoff was not adopted before runtime startup".into()));
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
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let flags = if enabled { flags | libc::FD_CLOEXEC } else { flags & !libc::FD_CLOEXEC };
    // SAFETY: the descriptor and flag word were checked above.
    if unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
