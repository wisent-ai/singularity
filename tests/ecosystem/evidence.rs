use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::os::unix::process::ExitStatusExt;
use std::{
    fs,
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub type Result<T> = std::result::Result<T, String>;
pub const BINARY: &str = env!("CARGO_BIN_EXE_singularity");
pub const SOURCE_REVISION: Option<&str> = option_env!("WISENT_SOURCE_COMMIT");

pub struct Evidence {
    pub root: PathBuf,
    document: Value,
}

impl Evidence {
    pub fn new() -> Result<Self> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".wisent-output/ecosystem-tests")
            .join(uuid::Uuid::new_v4().to_string());
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&root)
            .map_err(|e| e.to_string())?;
        let document = json!({
            "source_revision":SOURCE_REVISION,"binary":BINARY,"binary_sha256":file_digest(Path::new(BINARY))?,
            "working_directory":root,
            "started_at":chrono::Utc::now(),"outcome":"running","commands":[],"owner_attempts":[],"observations":{}
        });
        let value = Self { root, document };
        value.save()?;
        Ok(value)
    }

    pub fn command(&mut self, arguments: &[String]) -> Result<Output> {
        let output = Command::new(BINARY)
            .args(arguments)
            .current_dir(&self.root)
            .output()
            .map_err(|e| format!("execute {BINARY}: {e}"))?;
        self.document["commands"].as_array_mut().unwrap().push(json!({
            "executable":BINARY,"argv":arguments,"exit_code":output.status.code(),"signal":output.status.signal(),
            "stdout":String::from_utf8_lossy(&output.stdout),"stderr":String::from_utf8_lossy(&output.stderr)
        }));
        self.save()?;
        Ok(output)
    }

    pub fn snapshot(&mut self, name: &str, value: Value) -> Result<()> {
        self.document["observations"][name] = value;
        self.save()
    }

    pub fn owner_exit(
        &mut self,
        arguments: &[String],
        stdout: &Path,
        stderr: &Path,
        status: std::process::ExitStatus,
    ) -> Result<()> {
        self.document["owner_attempts"].as_array_mut().unwrap().push(json!({
            "executable":BINARY,"argv":arguments,"exit_code":status.code(),"signal":status.signal(),
            "stdout":fs::read_to_string(stdout).map_err(|e| e.to_string())?,
            "stderr":fs::read_to_string(stderr).map_err(|e| e.to_string())?
        }));
        self.save()
    }

    fn save(&self) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.document).map_err(|e| e.to_string())?;
        let path = self.root.join("report.json");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)
            .map_err(|e| e.to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())
    }

    pub fn finish(&mut self, result: &Result<()>) -> Result<()> {
        self.document["outcome"] = json!(if result.is_ok() { "passed" } else { "failed" });
        self.document["failure"] = json!(result.as_ref().err());
        self.document["completed_at"] = json!(chrono::Utc::now());
        self.save()?;
        if let Some(directory) = std::env::var_os("WISENT_TEST_EVIDENCE_DIR") {
            let directory = PathBuf::from(directory);
            let name = self
                .root
                .file_name()
                .ok_or("evidence directory has no identity")?
                .to_string_lossy();
            fs::copy(
                self.root.join("report.json"),
                directory.join(format!("ecosystem-{name}.json")),
            )
            .map_err(|e| format!("retain Stado qualification report: {e}"))?;
        }
        println!(
            "Ecosystem qualification report: {}",
            self.root.join("report.json").display()
        );
        Ok(())
    }
}

pub fn file_digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
