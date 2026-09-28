use super::fleet::{metadata, reservation};
use super::{AppError, Result, Store, meta_text, parsed, set_meta_text, sql};
use rust_decimal::Decimal;
use sea_orm::{
    ActiveValue::Set, ColumnTrait, ConnectionTrait, DatabaseTransaction, EntityTrait, QueryFilter,
    QuerySelect, TransactionTrait,
};
use serde_json::json;

pub struct Settlement {
    reserved: Decimal,
    actual: Decimal,
}

impl Settlement {
    pub fn check(self, request: &str) -> Result<()> {
        if self.actual > self.reserved {
            return Err(AppError::State(format!(
                "provider exceeded reserved cost: request {request}, reserved {}, observed {}; the portfolio was paused",
                self.reserved, self.actual
            )));
        }
        Ok(())
    }
}

/// Spent and reserved dollars as the ledger holds them inside `db`'s view.
async fn ledger(db: &impl ConnectionTrait, being: &str) -> Result<(Decimal, Decimal)> {
    Ok((
        parsed(meta_text(db, being, "spent_usd").await?)?
            .ok_or_else(|| AppError::State("missing spend ledger".into()))?,
        parsed(meta_text(db, being, "reserved_usd").await?)?
            .ok_or_else(|| AppError::State("missing reservation ledger".into()))?,
    ))
}

/// Lock the being's reservation ledger row, so two reservations or
/// settlements of one being serialize.
async fn lock_ledger(tx: &DatabaseTransaction, being: &str) -> Result<()> {
    metadata::Entity::find_by_id((being.to_owned(), "reserved_usd".to_owned()))
        .lock_exclusive()
        .one(tx)
        .await
        .map_err(sql)?
        .ok_or_else(|| AppError::State("missing reservation ledger".into()))?;
    Ok(())
}

/// The being's reservation `id`, if any.
async fn reservation_of(
    tx: &DatabaseTransaction,
    being: &str,
    id: &str,
) -> Result<Option<reservation::Model>> {
    reservation::Entity::find_by_id((being.to_owned(), id.to_owned()))
        .one(tx)
        .await
        .map_err(sql)
}

fn decimal(text: &str, what: &str) -> Result<Decimal> {
    text.parse()
        .map_err(|error| AppError::State(format!("{what}: {error}")))
}

impl Store {
    pub fn balance(&self) -> Result<(Decimal, Decimal)> {
        let being = self.being.clone();
        self.db
            .run(move |db| async move { ledger(&db, &being).await })
    }
    pub fn settled_cost(&self, id: &str) -> Result<Option<Decimal>> {
        let key = (self.being.clone(), id.to_owned());
        let amount = self.db.run(move |db| async move {
            Ok(reservation::Entity::find_by_id(key)
                .filter(reservation::Column::State.eq("settled"))
                .one(&db)
                .await
                .map_err(sql)?
                .and_then(|row| row.actual))
        })?;
        amount
            .map(|amount| decimal(&amount, &format!("recorded cost for {id}")))
            .transpose()
    }

    /// Reserve before sending. Unknown outcomes retain the reservation across a crash.
    pub fn reserve(&self, id: &str, amount: Decimal, limit: Decimal) -> Result<()> {
        if amount <= Decimal::ZERO {
            return Err(AppError::Config("reservation must be positive".into()));
        }
        let (being, id) = (self.being.clone(), id.to_owned());
        self.db.run(move |db| async move {
            let tx = db.begin().await.map_err(sql)?;
            lock_ledger(&tx, &being).await?;
            if let Some(row) = reservation_of(&tx, &being, &id).await? {
                let previous = decimal(&row.amount, "reservation amount")?;
                if previous == amount && row.state == "reserved" {
                    return Ok(());
                }
                return Err(AppError::State(format!(
                    "reservation {id} conflicts with its original amount or settlement"
                )));
            }
            let (spent, reserved) = ledger(&tx, &being).await?;
            let next_reserved = reserved.checked_add(amount).ok_or_else(|| {
                AppError::State("reservation exceeds the ledger's decimal range".into())
            })?;
            let allocated = spent.checked_add(next_reserved).ok_or_else(|| {
                AppError::State("total allocation exceeds the ledger's decimal range".into())
            })?;
            if allocated > limit {
                return Err(AppError::State(
                    "budget exhausted: existing reservations are not available for reuse".into(),
                ));
            }
            reservation::Entity::insert(reservation::ActiveModel {
                being: Set(being.clone()),
                id: Set(id),
                amount: Set(amount.to_string()),
                state: Set("reserved".into()),
                actual: Set(None),
            })
            .exec_without_returning(&tx)
            .await
            .map_err(sql)?;
            let value = json!(next_reserved.to_string()).to_string();
            set_meta_text(&tx, &being, "reserved_usd", &value).await?;
            tx.commit().await.map_err(sql)
        })
    }
    /// Committing an overrun also closes admission; a crash cannot separate these effects.
    pub fn settle(&self, id: &str, actual: Decimal) -> Result<Settlement> {
        if actual < Decimal::ZERO {
            return Err(AppError::State("negative reported cost".into()));
        }
        let (being, id) = (self.being.clone(), id.to_owned());
        self.db.run(move |db| async move {
            let tx = db.begin().await.map_err(sql)?;
            lock_ledger(&tx, &being).await?;
            let row = reservation_of(&tx, &being, &id)
                .await?
                .ok_or_else(|| AppError::State(format!("reservation {id} is unknown")))?;
            let amount = decimal(&row.amount, "reservation amount")?;
            if row.state == "settled" {
                let previous = decimal(
                    &row.actual.ok_or_else(|| {
                        AppError::State("settled reservation has no recorded cost".into())
                    })?,
                    "recorded settlement",
                )?;
                if previous != actual {
                    return Err(AppError::State(
                        "conflicting repeated cost settlement".into(),
                    ));
                }
            } else if row.state == "reserved" {
                let (spent, reserved) = ledger(&tx, &being).await?;
                let spent = spent.checked_add(actual).ok_or_else(|| {
                    AppError::State("observed spend exceeds the ledger's decimal range".into())
                })?;
                let reserved = reserved
                    .checked_sub(amount)
                    .filter(|value| *value >= Decimal::ZERO)
                    .ok_or_else(|| {
                        AppError::State("reservation exceeds the retained allocation ledger".into())
                    })?;
                reservation::Entity::update(reservation::ActiveModel {
                    being: Set(being.clone()),
                    id: Set(id.clone()),
                    state: Set("settled".into()),
                    actual: Set(Some(actual.to_string())),
                    ..Default::default()
                })
                .exec(&tx)
                .await
                .map_err(sql)?;
                for (key, value) in [("spent_usd", spent), ("reserved_usd", reserved)] {
                    set_meta_text(&tx, &being, key, &json!(value.to_string()).to_string()).await?;
                }
            } else {
                let state = row.state;
                return Err(AppError::State(format!(
                    "reservation {id} has unsupported state {state}"
                )));
            }
            if actual > amount {
                set_meta_text(&tx, &being, "paused", "true").await?;
            }
            tx.commit().await.map_err(sql)?;
            Ok(Settlement {
                reserved: amount,
                actual,
            })
        })
    }
}
