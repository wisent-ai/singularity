# Singularity Agent Memory

## Last Session (#13)
- **Built**: DeploymentSkill (PR #115) — cloud deployment management for Docker, fly.io, Railway, and local environments. 9 actions: generate_config, deploy, status, list_deployments, scale, rollback, destroy, get_deploy_history, generate_dockerfile
- **Tests**: 20 tests pass, all 17 smoke tests pass
- **Pillar**: Replication — replicas can now actually be deployed somewhere

## Previous PRs
- PR #115: DeploymentSkill — cloud deployment management with config generation, scaling, rollback
- PR #114: ContextSynthesisSkill — cross-skill context aggregation
- PR #113: SecretVaultSkill — secure credential management with encryption, access control, expiry
- PR #112: AgentNetworkSkill — service discovery, capability routing, and RPC
- PR #111: PaymentSkill — invoice management, payment processing, revenue tracking
- PR #109: AgentHealthMonitor — replica fleet health checking and alerting
- PR #108: WebhookSkill — inbound webhook endpoint management
- PR #107: AutonomousLoopSkill — central executive for autonomous operation
- PR #106: SkillComposerSkill — dynamic skill composition and code generation
- PR #105: UsageTrackingSkill — per-customer API usage metering and billing
- PR #104: SessionBootstrapSkill — unified session startup orchestration
- PR #103: AutoCatalogSkill — complete revenue pipeline
- PR #102: ResourceWatcherSkill — budget monitoring and cost optimization
- PR #101: RevenueServiceSkill — concrete value-producing services

## Current Skill Count: ~52 skills

## What to Build Next (Priority Order)
1. **CredentialDiscoverySkill** — Auto-detect available API keys from environment variables and populate the SecretVault. Self-Improvement pillar — agent should know what it can do.
2. **ServiceMeshSkill** — Service routing and load balancing across agent replicas. Replication pillar — multiple agents need traffic distribution.
3. **BillingDashboardSkill** — Generate HTML billing dashboards for customers. Revenue pillar — customers need visibility into their usage.
4. **PromptChainSkill** — Multi-step prompt orchestration with branching logic. Self-Improvement pillar — more complex reasoning chains.
5. **EnvironmentReplicatorSkill** — Clone current environment config (installed skills, settings, credentials manifest) for bootstrapping new replicas. Replication pillar — replicas should start with parent's capabilities.

## Architecture Notes
- All skills follow: Skill base class → SkillManifest → async execute(action, params) → SkillResult
- Data stored in singularity/data/*.json files
- Tests use tmp_path fixtures with mock.patch on file paths
- Skills registered in autonomous_agent.py DEFAULT_SKILL_CLASSES list
