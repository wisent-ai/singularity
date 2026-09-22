mod accounting;
mod initialization;
mod queries;
mod records;
use super::model::{Initiative, Issue, Policy};
use crate::AppError;
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type Result<T, E = AppError> = std::result::Result<T, E>;
fn sql(error: rusqlite::Error) -> AppError {
    AppError::State(format!("ecosystem database: {error}"))
}

pub struct Store {
    connection: Connection,
}
impl Store {
    pub fn meta<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let value: Option<String> = self
            .connection
            .query_row("SELECT value FROM metadata WHERE key=?1", [key], |r| {
                r.get(0)
            })
            .optional()
            .map_err(sql)?;
        value
            .map(|v| serde_json::from_str(&v).map_err(AppError::from))
            .transpose()
    }
    pub fn set_meta(&self, key: &str, value: &impl Serialize) -> Result<()> {
        self.connection.execute("INSERT INTO metadata VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, serde_json::to_string(value)?]).map_err(sql)?;
        Ok(())
    }
    pub fn clear_meta(&self, key: &str) -> Result<()> {
        self.connection
            .execute("DELETE FROM metadata WHERE key=?1", [key])
            .map_err(sql)?;
        Ok(())
    }
    pub fn get<T: DeserializeOwned>(&self, kind: &str, id: &str) -> Result<Option<T>> {
        let value: Option<String> = self
            .connection
            .query_row(
                "SELECT data FROM records WHERE kind=?1 AND id=?2",
                params![kind, id],
                |r| r.get(0),
            )
            .optional()
            .map_err(sql)?;
        value
            .map(|v| serde_json::from_str(&v).map_err(AppError::from))
            .transpose()
    }
    pub fn list<T: DeserializeOwned>(&self, kind: &str) -> Result<Vec<T>> {
        let mut statement = self
            .connection
            .prepare("SELECT data FROM records WHERE kind=?1 ORDER BY created_at,id")
            .map_err(sql)?;
        statement
            .query_map([kind], |r| r.get::<_, String>(0))
            .map_err(sql)?
            .map(|row| serde_json::from_str(&row.map_err(sql)?).map_err(AppError::from))
            .collect()
    }
    pub fn put(
        &self,
        kind: &str,
        id: &str,
        value: &impl Serialize,
        initiative: Option<&str>,
        detail: &str,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let data = serde_json::to_string(value)?;
        let content_sha256 = format!("{:x}", Sha256::digest(data.as_bytes()));
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        tx.execute("INSERT INTO records (kind,id,data,created_at,updated_at,content_sha256) VALUES (?1,?2,?3,?4,?4,?5) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data,updated_at=excluded.updated_at,content_sha256=excluded.content_sha256",
            params![kind,id,data,now,content_sha256]).map_err(sql)?;
        let event_id = format!("event-{}", uuid::Uuid::new_v4());
        let event = json!({"id":event_id,"initiative_id":initiative,"kind":kind,"record_id":id,"record_sha256":content_sha256,"detail":detail,"created_at":now}).to_string();
        let event_sha256 = format!("{:x}", Sha256::digest(event.as_bytes()));
        tx.execute(
            "INSERT INTO records(kind,id,data,created_at,updated_at,content_sha256) VALUES ('event',?1,?2,?3,?3,?4)",
            params![event_id,event,now,event_sha256],
        ).map_err(sql)?;
        tx.execute("INSERT INTO metadata VALUES ('last_progress_at',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [json!(now).to_string()]).map_err(sql)?;
        tx.commit().map_err(sql)?;
        Ok(())
    }
    pub fn issue(&self, issue: &Issue) -> Result<()> {
        self.put("issue", &issue.operation, issue, None, &issue.message)
    }
    pub fn clear_issue(&self, operation: &str) -> Result<()> {
        self.connection
            .execute(
                "DELETE FROM records WHERE kind='issue' AND id=?1",
                [operation],
            )
            .map_err(sql)?;
        Ok(())
    }
    pub fn paused(&self) -> Result<bool> {
        self.meta("paused")?
            .ok_or_else(|| AppError::State("missing pause state".into()))
    }
    pub fn status(&self, policy: &Policy) -> Result<Value> {
        let (spent, reserved) = self.balance()?;
        let count = |kind| -> Result<i64> {
            self.connection
                .query_row("SELECT count(*) FROM records WHERE kind=?1", [kind], |r| {
                    r.get(0)
                })
                .map_err(sql)
        };
        Ok(json!({"paused":self.paused()?,"updated_at":Utc::now(),
            "las_catalog_ready":self.meta::<bool>("las_catalog_ready")?.unwrap_or(false),
            "owner":self.meta::<Value>("owner")?,
            "runtime":{"version":env!("CARGO_PKG_VERSION"),"source_revision":option_env!("WISENT_SOURCE_COMMIT")},
            "last_progress_at":self.meta::<String>("last_progress_at")?,
            "active_count":self.active_count()?,
            "observation_count":count("observation")?,"opportunity_count":count("opportunity")?,"initiative_count":count("initiative")?,
            "spent_usd":spent.to_string(),"reserved_usd":reserved.to_string(),"budget_usd":policy.budget_usd.to_string(),
            "issues":self.list::<Issue>("issue")?}))
    }
    pub fn explain(&self, id: &str) -> Result<Value> {
        let initiative: Initiative = self
            .get("initiative", id)?
            .ok_or_else(|| AppError::State(format!("unknown initiative {id}")))?;
        let mut statement = self
            .connection
            .prepare(
                "SELECT kind,count(*) FROM records WHERE json_extract(data,'$.initiative_id')=?1 GROUP BY kind ORDER BY kind",
            )
            .map_err(sql)?;
        let related = statement
            .query_map([id], |row| {
                Ok(json!({"kind":row.get::<_,String>(0)?,"count":row.get::<_,i64>(1)?}))
            })
            .map_err(sql)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sql)?;
        Ok(
            json!({"opportunity":self.get::<Value>("opportunity",&initiative.opportunity_id)?,
                "review":self.get::<Value>("review",&initiative.opportunity_id)?,"initiative":initiative,
                "related":related,"history":{"method":"records","params":{"initiative_id":id,"limit":super::protocol::DEFAULT_PAGE_SIZE}}}),
        )
    }
}
