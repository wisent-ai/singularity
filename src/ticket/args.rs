//! The flags of `singularity ticket key|sign`. Every value the ticket carries
//! is given explicitly; nothing is read from the environment or defaulted.

use std::path::PathBuf;

use clap::{Args, Subcommand};

/// `singularity ticket`: write the keys and the signed launch ticket
/// `singularity-bootstrap` starts a managed being from.
#[derive(Debug, Clone, Args)]
pub struct TicketArgs {
    #[command(subcommand)]
    pub verb: TicketVerb,
}

#[derive(Debug, Clone, Subcommand)]
pub enum TicketVerb {
    /// Write a fresh Ed25519 key pair: a supervisor's (its public half is the trust root) or a workload's (its public half is what skarbiec grant issue --workload-public-key-file registers)
    Key(KeyArgs),
    /// Write one singularity.bootstrap.v2 manifest and its signature from explicit inputs, then check it the way singularity-bootstrap will
    Sign(SignArgs),
}

#[derive(Debug, Clone, Args)]
pub struct KeyArgs {
    /// Absolute path of the private key; refused when it exists
    #[arg(long)]
    pub out: PathBuf,
    /// Absolute path of the public key: the trust root of a supervisor key, or the file skarbiec grant issue --workload-public-key-file reads for a workload key
    #[arg(long)]
    pub public_out: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct SignArgs {
    #[arg(long)]
    pub agent_id: String,
    #[arg(long)]
    pub role: String,
    #[arg(long)]
    pub environment: String,
    #[arg(long)]
    pub host: String,
    /// The workload id the three capabilities were issued to
    #[arg(long)]
    pub workload_id: String,
    /// The workload's private key file (from `singularity ticket key`); its public half goes into the ticket
    #[arg(long)]
    pub workload_key: PathBuf,
    /// The Skarbiec broker socket the capabilities are redeemed through
    #[arg(long)]
    pub broker_socket: PathBuf,
    /// The singularity executable the bootstrap execs; its SHA-256 goes into the ticket
    #[arg(long)]
    pub executable: PathBuf,
    /// The code digest the runtime reports (lowercase SHA-256 hex)
    #[arg(long)]
    pub code_digest: String,
    /// The ecosystem policy file; its SHA-256 goes into the ticket
    #[arg(long)]
    pub policy_file: PathBuf,
    /// The policy sequence, greater than the last one this being ran under
    #[arg(long)]
    pub policy_sequence: u64,
    /// How long the ticket is valid from now
    #[arg(long)]
    pub expires_in_seconds: u32,
    /// Capability id issued for the Brama HMAC (purpose singularity.brama.bootstrap)
    #[arg(long)]
    pub brama_capability: String,
    /// Its resource, brama:<field>
    #[arg(long)]
    pub brama_resource: String,
    /// Capability id issued for the Brama bearer (purpose singularity.brama.authorization)
    #[arg(long)]
    pub brama_bearer_capability: String,
    /// Its resource, brama:<field>
    #[arg(long)]
    pub brama_bearer_resource: String,
    /// Capability id issued for the Most token (purpose singularity.most.bootstrap)
    #[arg(long)]
    pub most_capability: String,
    /// Its resource, most:<field>
    #[arg(long)]
    pub most_resource: String,
    /// The supervisor's private key file
    #[arg(long)]
    pub supervisor_key: PathBuf,
    /// The supervisor's public key file singularity-bootstrap is given as --trust-root
    #[arg(long)]
    pub trust_root: PathBuf,
    /// Absolute path of the manifest to write; refused when it exists
    #[arg(long)]
    pub manifest_out: PathBuf,
    /// Absolute path of the signature to write; refused when it exists
    #[arg(long)]
    pub signature_out: PathBuf,
    /// Arguments singularity is started with, after `--`
    #[arg(last = true)]
    pub singularity_args: Vec<String>,
}
