//! Where the ecosystem store lives: the fleet database `singularity`,
//! reached the one way every Wisent product reaches its database —
//! `stado_database` (Stado resolve, the Skarbiec route, the
//! `singularity-database-client` bearer) — as a SeaORM connection. The
//! tables are the entities below; the migrator creates them.
//!
//! The store is called from synchronous code, so it uses Stado's shared
//! synchronous client and hands it the store's entity work with `run`.

use super::{AppError, Result};
use sea_orm::DatabaseConnection;
use sea_orm_migration::prelude::*;
use std::future::Future;

/// `HOME` holding `.stado/` when the process's own `HOME` is isolated.
const STADO_HOME_VARIABLE: &str = "SINGULARITY_STADO_HOME";

/// Tables and indexes of the ecosystem store; `data` stays text so a record's
/// digest and byte offsets are those of the JSON the being wrote.
const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS ecosystem_metadata (being TEXT NOT NULL, key TEXT NOT NULL,
        value TEXT NOT NULL, PRIMARY KEY(being,key));
    CREATE TABLE IF NOT EXISTS ecosystem_records (seq BIGSERIAL UNIQUE, being TEXT NOT NULL,
        kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, content_sha256 TEXT NOT NULL, PRIMARY KEY(being,kind,id));
    CREATE TABLE IF NOT EXISTS ecosystem_reservations (being TEXT NOT NULL, id TEXT NOT NULL,
        amount TEXT NOT NULL, state TEXT NOT NULL, actual TEXT, PRIMARY KEY(being,id));
    CREATE INDEX IF NOT EXISTS ecosystem_observations_source_time ON ecosystem_records
        (being,(data::jsonb->>'source'),created_at DESC,id DESC) WHERE kind='observation';
    CREATE INDEX IF NOT EXISTS ecosystem_records_initiative_time ON ecosystem_records
        (being,kind,(data::jsonb->>'initiative_id'),created_at,id);
    CREATE INDEX IF NOT EXISTS ecosystem_records_state_time ON ecosystem_records
        (being,kind,(data::jsonb->>'state'),created_at,id);
    CREATE INDEX IF NOT EXISTS ecosystem_records_sequence ON ecosystem_records(being,seq);";

/// `ecosystem_metadata`: one JSON value of a being, by key.
pub(super) mod metadata {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "ecosystem_metadata")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub being: String,
        #[sea_orm(primary_key, auto_increment = false)]
        pub key: String,
        pub value: String,
    }
    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}
    impl ActiveModelBehavior for ActiveModel {}
}

/// `ecosystem_records`: one record of a being; `seq` orders the history.
pub(super) mod record {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "ecosystem_records")]
    pub struct Model {
        pub seq: i64,
        #[sea_orm(primary_key, auto_increment = false)]
        pub being: String,
        #[sea_orm(primary_key, auto_increment = false)]
        pub kind: String,
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub data: String,
        pub created_at: String,
        pub updated_at: String,
        pub content_sha256: String,
    }
    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}
    impl ActiveModelBehavior for ActiveModel {}
}

/// `ecosystem_reservations`: one reserved spend of a being and its outcome.
pub(super) mod reservation {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "ecosystem_reservations")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub being: String,
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub amount: String,
        pub state: String,
        pub actual: Option<String>,
    }
    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {}
    impl ActiveModelBehavior for ActiveModel {}
}

struct Migrator;

impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![Box::new(EcosystemTables)]
    }
}

struct EcosystemTables;

impl MigrationName for EcosystemTables {
    fn name(&self) -> &str {
        "m20260928_000001_ecosystem"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for EcosystemTables {
    async fn up(&self, manager: &SchemaManager) -> std::result::Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared(SCHEMA)
            .await
            .map(|_| ())
    }
}

fn failed(step: &str, detail: impl std::fmt::Display) -> AppError {
    AppError::State(format!("ecosystem database: {step}: {detail}"))
}

/// The being's connection: Stado's shared synchronous client, handed the
/// store's entity work with `run`.
pub(super) struct Db {
    client: stado_database::sync::Client,
}

impl Db {
    /// Resolve, connect and create the tables; a refusal names the step.
    pub(super) fn start() -> Result<Self> {
        let database =
            stado_database::FleetDatabase::for_product("singularity", STADO_HOME_VARIABLE)
                .map_err(|error| AppError::State(error.to_string()))?;
        let client = stado_database::sync::Client::connect(&database)
            .map_err(|error| AppError::State(error.to_string()))?;
        client
            .run(|db| async move { Migrator::up(&db, None).await })
            .map_err(|error| failed("creating the ecosystem tables", error))?
            .map_err(|error| failed("creating the ecosystem tables", error))?;
        Ok(Self { client })
    }

    /// Run `work` against the connection and wait for its answer.
    pub(super) fn run<R, F>(&self, work: impl FnOnce(DatabaseConnection) -> F) -> Result<R>
    where
        R: Send + 'static,
        F: Future<Output = Result<R>> + Send + 'static,
    {
        self.client
            .run(work)
            .map_err(|error| failed("running an operation", error))?
    }
}
