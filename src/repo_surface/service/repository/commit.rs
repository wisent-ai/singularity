use super::*;

/// A lost Git response can be reconciled only from the exact tree, parent and request marker.
pub(crate) async fn reconcile_commit(state: &WorkspaceState, message: &str, request: &str, sealed: &str) -> SurfaceResult<String> {
    let branch = successful(git(&state.worktree, &["branch", "--show-current"], None, 30).await?, "read commit branch")?;
    if branch.stdout.trim() != state.branch {
        return Err(SurfaceError::conflict("canonical branch changed before commit"));
    }
    let expected_message = format!("{message}\n\nWisent-Request: {request}");
    let head = successful(git(&state.worktree, &["rev-parse", "HEAD"], None, 30).await?, "read commit head")?;
    if head.stdout.trim() == state.base_commit {
        successful(git(&state.worktree, &["commit", "-m", &expected_message], None, 120).await?, "create canonical commit")?;
    }
    let tree = successful(git(&state.worktree, &["rev-parse", "HEAD^{tree}"], None, 30).await?, "read committed tree")?;
    let parent = successful(git(&state.worktree, &["rev-parse", "HEAD^"], None, 30).await?, "read committed parent")?;
    let body = successful(git(&state.worktree, &["log", "-1", "--format=%B"], None, 30).await?, "read committed request identity")?;
    if tree.stdout.trim() != sealed || parent.stdout.trim() != state.base_commit || body.stdout.trim_end() != expected_message {
        return Err(SurfaceError::conflict("commit outcome does not match the sealed tree, original parent and immutable request; no commit was replayed"));
    }
    Ok(successful(git(&state.worktree, &["rev-parse", "HEAD"], None, 30).await?, "resolve canonical commit")?.stdout.trim().to_owned())
}
