use super::*;

impl StateStore {
    pub fn existing_workspace(&self, id: &str) -> SurfaceResult<Option<WorkspaceState>> {
        let path = self.record_path(id)?;
        match fs::symlink_metadata(&path) {
            Ok(_) => self.load_workspace(id).map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(SurfaceError::state(format!("cannot inspect workspace {id}: {error}"))),
        }
    }

    pub fn claim_repository(&self, repository: &str, workspace: &str) -> SurfaceResult<()> {
        validate_id("repository id", repository)?;
        validate_id("workspace id", workspace)?;
        let _lock = locking::acquire(&self.root.join("repository-locks").join(format!("{repository}.lock")))?;
        let path = self.root.join("repositories").join(format!("{repository}.json"));
        if path.exists() {
            let previous: String = read_owner_json(&path, "repository owner")?;
            if previous == workspace { return Ok(()); }
            let previous = self.load_workspace(&previous)?;
            if !previous.published {
                return Err(SurfaceError::conflict(format!("repository {repository} is owned by unfinished workspace {}", previous.id)));
            }
            atomic_owner_json(&path, &workspace)
        } else {
            atomic_owner_json_new(&path, &workspace)
        }
    }

    pub fn require_repository_owner(&self, repository: &str, workspace: &str) -> SurfaceResult<()> {
        validate_id("repository id", repository)?;
        let path = self.root.join("repositories").join(format!("{repository}.json"));
        let owner: String = read_owner_json(&path, "repository owner")?;
        if owner != workspace {
            return Err(SurfaceError::conflict("workspace no longer owns the canonical repository"));
        }
        Ok(())
    }
}
