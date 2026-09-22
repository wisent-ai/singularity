use super::{AppError, Result, Store, sql};
use rusqlite::{OptionalExtension, params};
use rust_decimal::Decimal;
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

impl Store {
    pub fn balance(&self) -> Result<(Decimal, Decimal)> {
        Ok((
            self.meta("spent_usd")?
                .ok_or_else(|| AppError::State("missing spend ledger".into()))?,
            self.meta("reserved_usd")?
                .ok_or_else(|| AppError::State("missing reservation ledger".into()))?,
        ))
    }
    pub fn settled_cost(&self, id: &str) -> Result<Option<Decimal>> {
        let amount: Option<String> = self
            .connection
            .query_row(
                "SELECT actual FROM reservations WHERE id=?1 AND state='settled'",
                [id],
                |row| row.get(0),
            )
            .optional()
            .map_err(sql)?;
        amount
            .map(|amount| {
                amount
                    .parse()
                    .map_err(|error| AppError::State(format!("recorded cost for {id}: {error}")))
            })
            .transpose()
    }

    /// Reserve before sending. Unknown outcomes retain the reservation across a crash.
    pub fn reserve(&self, id: &str, amount: Decimal, limit: Decimal) -> Result<()> {
        if amount <= Decimal::ZERO {
            return Err(AppError::Config("reservation must be positive".into()));
        }
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let existing: Option<(String, String)> = tx
            .query_row(
                "SELECT amount,state FROM reservations WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(sql)?;
        if let Some((previous, state)) = existing {
            let previous: Decimal = previous
                .parse()
                .map_err(|error| AppError::State(format!("reservation amount: {error}")))?;
            if previous == amount && state == "reserved" {
                return Ok(());
            }
            return Err(AppError::State(format!(
                "reservation {id} conflicts with its original amount or settlement"
            )));
        }
        let (spent, reserved) = self.balance()?;
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
        tx.execute(
            "INSERT INTO reservations VALUES (?1,?2,'reserved',NULL)",
            params![id, amount.to_string()],
        )
        .map_err(sql)?;
        tx.execute(
            "UPDATE metadata SET value=?1 WHERE key='reserved_usd'",
            [json!(next_reserved.to_string()).to_string()],
        )
        .map_err(sql)?;
        tx.commit().map_err(sql)?;
        Ok(())
    }
    /// Committing an overrun also closes admission; a crash cannot separate these effects.
    pub fn settle(&self, id: &str, actual: Decimal) -> Result<Settlement> {
        if actual < Decimal::ZERO {
            return Err(AppError::State("negative reported cost".into()));
        }
        let tx = self.connection.unchecked_transaction().map_err(sql)?;
        let (amount, state, old): (String, String, Option<String>) = tx
            .query_row(
                "SELECT amount,state,actual FROM reservations WHERE id=?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(sql)?;
        let amount: Decimal = amount
            .parse()
            .map_err(|error| AppError::State(format!("reservation amount: {error}")))?;
        if state == "settled" {
            let previous: Decimal = old
                .ok_or_else(|| AppError::State("settled reservation has no recorded cost".into()))?
                .parse()
                .map_err(|error| AppError::State(format!("recorded settlement: {error}")))?;
            if previous != actual {
                return Err(AppError::State(
                    "conflicting repeated cost settlement".into(),
                ));
            }
        } else if state == "reserved" {
            let (spent, reserved) = self.balance()?;
            let spent = spent.checked_add(actual).ok_or_else(|| {
                AppError::State("observed spend exceeds the ledger's decimal range".into())
            })?;
            let reserved = reserved
                .checked_sub(amount)
                .filter(|value| *value >= Decimal::ZERO)
                .ok_or_else(|| {
                    AppError::State("reservation exceeds the retained allocation ledger".into())
                })?;
            tx.execute(
                "UPDATE reservations SET state='settled',actual=?1 WHERE id=?2",
                params![actual.to_string(), id],
            )
            .map_err(sql)?;
            for (key, value) in [("spent_usd", spent), ("reserved_usd", reserved)] {
                tx.execute(
                    "UPDATE metadata SET value=?1 WHERE key=?2",
                    params![json!(value.to_string()).to_string(), key],
                )
                .map_err(sql)?;
            }
        } else {
            return Err(AppError::State(format!(
                "reservation {id} has unsupported state {state}"
            )));
        }
        if actual > amount {
            tx.execute("UPDATE metadata SET value='true' WHERE key='paused'", [])
                .map_err(sql)?;
        }
        tx.commit().map_err(sql)?;
        Ok(Settlement {
            reserved: amount,
            actual,
        })
    }
}
