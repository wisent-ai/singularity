use super::super::{Shared, observe};
use crate::AppError;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::{
    fs,
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

pub fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 100
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && value.as_bytes()[0].is_ascii_alphanumeric()
}

pub fn immutable_request(directory: &Path, id: &str, request: &Value) -> Result<PathBuf, AppError> {
    if !identifier(id) {
        return Err(AppError::Config(
            "invalid execution request identifier".into(),
        ));
    }
    let directory = directory.join("requests");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{id}.json"));
    let bytes = serde_json::to_vec_pretty(request)?;
    let pending = directory.join(format!(".{id}.pending"));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&pending)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    let published = fs::hard_link(&pending, &path);
    fs::remove_file(&pending)?;
    match published {
        Ok(()) => fs::File::open(&directory)?.sync_all()?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::symlink_metadata(&path)?.file_type().is_symlink() || fs::read(&path)? != bytes {
                return Err(AppError::State(format!(
                    "request {id} conflicts with its persisted payload"
                )));
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(path)
}

pub async fn product(shared: &Shared, id: &str) -> Result<Value, AppError> {
    let catalog = observe::products(shared, &["catalog", "--json"]).await?;
    let product = catalog["products"]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["id"].as_str() == Some(id)))
        .cloned()
        .ok_or_else(|| {
            AppError::State(format!("Wisent Products has no registered product {id}"))
        })?;
    let state = shared.lock()?;
    state.store.put(
        "catalog_read",
        id,
        &product,
        None,
        "Read current product identity from Wisent Products",
    )?;
    Ok(product)
}

pub async fn checkout(shared: &Shared, product: &Value) -> Result<PathBuf, AppError> {
    let surfaces = product["surfaces"]
        .as_array()
        .ok_or_else(|| AppError::State("product has no surfaces".into()))?;
    let repository = surfaces
        .iter()
        .find(|s| s["kind"] == "cli")
        .or_else(|| surfaces.iter().find(|s| s["kind"] == "service"))
        .or_else(|| surfaces.first())
        .and_then(|s| s["repository"].as_str())
        .ok_or_else(|| AppError::State("product has no source repository".into()))?;
    let parts = repository.split('/').collect::<Vec<_>>();
    if parts.len() != 2 || !identifier(parts[1]) {
        return Err(AppError::State(
            "catalog repository is not an owner/name identity".into(),
        ));
    }
    let root = shared.lock()?.policy.workspace_root.clone();
    let path = fs::canonicalize(root.join(parts[1]))?;
    if path.parent() != Some(root.as_path()) {
        return Err(AppError::State(
            "repository escapes the canonical checkout root".into(),
        ));
    }
    let origin = git(&path, &["config", "--get", "remote.origin.url"]).await?;
    let normalized = origin.trim().trim_end_matches(".git");
    if normalized != format!("https://github.com/{repository}")
        && normalized != format!("git@github.com:{repository}")
    {
        return Err(AppError::State(format!(
            "checkout {} belongs to {origin}, not {repository}",
            path.display()
        )));
    }
    if git(&path, &["branch", "--show-current"]).await?.trim() != "main" {
        return Err(AppError::State(
            "canonical checkout is not on main; no branch was changed".into(),
        ));
    }
    let worktrees = git(&path, &["worktree", "list", "--porcelain"]).await?;
    if worktrees
        .lines()
        .filter(|line| line.starts_with("worktree "))
        .count()
        != 1
    {
        return Err(AppError::State(
            "repository has more than one checkout; no checkout was removed".into(),
        ));
    }
    Ok(path)
}

pub async fn git(path: &Path, args: &[&str]) -> Result<String, AppError> {
    let path = path
        .to_str()
        .ok_or_else(|| AppError::Config("checkout path is not UTF-8".into()))?;
    let mut argv = vec!["-C", path];
    argv.extend_from_slice(args);
    observe::text("git", &argv).await
}

fn evidence<T: DeserializeOwned>(run: &Path, value: &Value) -> Result<T, AppError> {
    let path =
        Path::new(value.as_str().ok_or_else(|| {
            AppError::State("execution response is missing an evidence path".into())
        })?)
        .canonicalize()?;
    if !path.starts_with(run) {
        return Err(AppError::State(
            "execution evidence is outside its run directory".into(),
        ));
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub fn accepted_receipt(response: &Value) -> Result<(), AppError> {
    let run = Path::new(
        response["run_directory"]
            .as_str()
            .ok_or_else(|| AppError::State("execution response has no run directory".into()))?,
    )
    .canonicalize()?;
    let contract: pursuit::TaskContract = evidence(&run, &response["contract"])?;
    let verdict: pursuit::TaskVerdict = evidence(&run, &response["verdict"])?;
    let receipt: pursuit::RunReceipt = evidence(&run, &response["receipt"])?;
    pursuit::validate_contract(&contract).map_err(AppError::State)?;
    pursuit::validate_verdict(&verdict, &contract).map_err(AppError::State)?;
    if receipt.state != "succeeded"
        || Some(receipt.run_id.as_str()) != response["run_id"].as_str()
        || !pursuit::verdict_accepted(&verdict)
    {
        return Err(AppError::State(
            "Pursuit did not record independent acceptance for this exact run".into(),
        ));
    }
    Ok(())
}
