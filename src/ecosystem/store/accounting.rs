use super::{AppError, Result, Store, meta_text, parsed, set_meta_text, sql};
use postgres::GenericClient;
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

/// Spent and reserved dollars as the ledger holds them inside `client`'s view.
fn ledger(client: &mut impl GenericClient, being: &str) -> Result<(Decimal, Decimal)> {
    Ok((
        parsed(meta_text(client, being, "spent_usd")?)?
            .ok_or_else(|| AppError::State("missing spend ledger".into()))?,
        parsed(meta_text(client, being, "reserved_usd")?)?
            .ok_or_else(|| AppError::State("missing reservation ledger".into()))?,
    ))
}

fn decimal(text: &str, what: &str) -> Result<Decimal> {
    text.parse()
        .map_err(|error| AppError::State(format!("{what}: {error}")))
}

impl Store {
    pub fn balance(&self) -> Result<(Decimal, Decimal)> {
        let being = self.being.clone();
        self.db.run(move |client| ledger(client, &being))
    }
    pub fn settled_cost(&self, id: &str) -> Result<Option<Decimal>> {
        let (being, id_owned) = (self.being.clone(), id.to_owned());
        let amount: Option<Option<String>> = self.db.run(move |client| {
            Ok(client
                .query_opt(
                    "SELECT actual FROM ecosystem_reservations WHERE being=$1 AND id=$2 AND state='settled'",
                    &[&being, &id_owned],
                )
                .map_err(sql)?
                .map(|row| row.get(0)))
        })?;
        amount
            .flatten()
            .map(|amount| decimal(&amount, &format!("recorded cost for {id}")))
            .transpose()
    }

    /// Reserve before sending. Unknown outcomes retain the reservation across a crash.
    pub fn reserve(&self, id: &str, amount: Decimal, limit: Decimal) -> Result<()> {
        if amount <= Decimal::ZERO {
            return Err(AppError::Config("reservation must be positive".into()));
        }
        let (being, id) = (self.being.clone(), id.to_owned());
        self.db.run(move |client| {
            let mut tx = client.transaction().map_err(sql)?;
            // The ledger row is locked first, so two reservations of one being serialize.
            tx.query_one(
                "SELECT value FROM ecosystem_metadata WHERE being=$1 AND key='reserved_usd' FOR UPDATE",
                &[&being],
            )
            .map_err(sql)?;
            let existing = tx
                .query_opt(
                    "SELECT amount,state FROM ecosystem_reservations WHERE being=$1 AND id=$2",
                    &[&being, &id],
                )
                .map_err(sql)?;
            if let Some(row) = existing {
                let previous = decimal(row.get(0), "reservation amount")?;
                if previous == amount && row.get::<_, &str>(1) == "reserved" {
                    return Ok(());
                }
                return Err(AppError::State(format!(
                    "reservation {id} conflicts with its original amount or settlement"
                )));
            }
            let (spent, reserved) = ledger(&mut tx, &being)?;
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
                "INSERT INTO ecosystem_reservations(being,id,amount,state,actual) VALUES ($1,$2,$3,'reserved',NULL)",
                &[&being, &id, &amount.to_string()],
            )
            .map_err(sql)?;
            set_meta_text(&mut tx, &being, "reserved_usd", &json!(next_reserved.to_string()).to_string())?;
            tx.commit().map_err(sql)
        })
    }
    /// Committing an overrun also closes admission; a crash cannot separate these effects.
    pub fn settle(&self, id: &str, actual: Decimal) -> Result<Settlement> {
        if actual < Decimal::ZERO {
            return Err(AppError::State("negative reported cost".into()));
        }
        let (being, id) = (self.being.clone(), id.to_owned());
        self.db.run(move |client| {
            let mut tx = client.transaction().map_err(sql)?;
            tx.query_one(
                "SELECT value FROM ecosystem_metadata WHERE being=$1 AND key='reserved_usd' FOR UPDATE",
                &[&being],
            )
            .map_err(sql)?;
            let row = tx
                .query_opt(
                    "SELECT amount,state,actual FROM ecosystem_reservations WHERE being=$1 AND id=$2",
                    &[&being, &id],
                )
                .map_err(sql)?
                .ok_or_else(|| AppError::State(format!("reservation {id} is unknown")))?;
            let amount = decimal(row.get(0), "reservation amount")?;
            let state: String = row.get(1);
            let old: Option<String> = row.get(2);
            if state == "settled" {
                let previous = decimal(
                    &old.ok_or_else(|| AppError::State("settled reservation has no recorded cost".into()))?,
                    "recorded settlement",
                )?;
                if previous != actual {
                    return Err(AppError::State(
                        "conflicting repeated cost settlement".into(),
                    ));
                }
            } else if state == "reserved" {
                let (spent, reserved) = ledger(&mut tx, &being)?;
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
                    "UPDATE ecosystem_reservations SET state='settled',actual=$3 WHERE being=$1 AND id=$2",
                    &[&being, &id, &actual.to_string()],
                )
                .map_err(sql)?;
                for (key, value) in [("spent_usd", spent), ("reserved_usd", reserved)] {
                    set_meta_text(&mut tx, &being, key, &json!(value.to_string()).to_string())?;
                }
            } else {
                return Err(AppError::State(format!(
                    "reservation {id} has unsupported state {state}"
                )));
            }
            if actual > amount {
                set_meta_text(&mut tx, &being, "paused", "true")?;
            }
            tx.commit().map_err(sql)?;
            Ok(Settlement {
                reserved: amount,
                actual,
            })
        })
    }
}
