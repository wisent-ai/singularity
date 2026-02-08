# Singularity Agent Memory

## Session 22 - BudgetAwarePlannerSkill (2026-02-08)

### What I Built
- **BudgetAwarePlannerSkill** (PR #138, merged) - Budget-constrained goal planning, the #1 priority from previous sessions
- 8 actions: estimate_goal, affordable_goals, plan_budget, record_cost, set_budget, roi_report, budget_status, learn_costs
- Per-pillar budget allocation (revenue 30%, self_improvement 25%, replication 20%, goal_setting 15%, other 10%)
- Confidence-weighted cost estimation with safety margins (low/medium/high confidence)
- Value-per-dollar scoring using greedy knapsack algorithm for goal prioritization
- ROI tracking per goal with estimation accuracy self-improvement
- 10% safety reserve to prevent budget exhaustion
- 18 tests pass, all 17 smoke tests pass

### Open Feature Requests
- None currently open. Check `gh issue list --label "feature-request" --state open`

### Architecture Notes
- Skills are auto-discovered by SkillLoader from singularity/skills/ directory
- All skills inherit from Skill base class in skills/base.py
- Must implement `manifest` as a @property (not get_manifest method)
- Must implement `async execute(self, action, params)` method
- Data persisted in singularity/data/*.json files
- SkillContext enables cross-skill communication
- service_api.py provides the FastAPI REST interface
- Messaging endpoints use /api/messages/* prefix, standalone skill creation if no agent

### Current State of Each Pillar

**Self-Improvement** (Strong)
- FeedbackLoopSkill, LearnedBehaviorSkill, PromptEvolutionSkill, SkillComposerSkill
- SkillDependencyAnalyzer for codebase introspection
- WorkflowAnalyticsSkill for pattern analysis
- DashboardSkill for self-awareness
- SelfTestingSkill, ErrorRecoverySkill
- SkillPerformanceProfiler for skill portfolio optimization
- CostAwareLLMRouter for model cost optimization

**Revenue Generation** (Strong)
- RevenueServicesSkill (5 value-producing services)
- UsageTrackingSkill (per-customer metering/billing)
- PaymentSkill, MarketplaceSkill, AutoCatalogSkill
- ServiceHostingSkill (HTTP service hosting for agents)
- ServiceAPI (FastAPI REST interface)
- AgentFundingSkill - grants, bounties, peer lending, contribution rewards
- CostOptimizerSkill - cost tracking and profitability analysis

**Replication** (Good)
- PeerDiscoverySkill, AgentNetworkSkill, AgentHealthMonitor
- DeploymentSkill (Docker/fly.io/Railway)
- KnowledgeSharingSkill
- MessagingSkill - agent-to-agent direct communication with REST API
- AgentFundingSkill - bootstrap funding for new replicas

**Goal Setting** (Moderate → Good)
- AutonomousLoopSkill, SessionBootstrapSkill
- GoalManager, Strategy, Planner skills
- DashboardSkill pillar scoring for priority decisions
- DecisionLogSkill for structured decision logging
- **BudgetAwarePlannerSkill** (NEW) - budget-constrained goal planning with ROI tracking

### What to Build Next
Priority order:
1. **Self-Healing Skill** - Detect failing subsystems and automatically restart/repair them
2. **Skill Marketplace** - Let agents list their skills for other agents to install/buy
3. **Webhook-Triggered Autonomous Workflows** - Connect WebhookSkill to AutonomousLoop
4. **API Gateway Skill** - Expose service_api.py as deployable endpoint with proper auth and rate limiting
5. **Task Delegation via AgentNetwork** - Parent spawns child with specific task and budget
6. **Goal Dependency Graph Visualizer** - Help agents understand goal relationships

### Key Files
- `singularity/skills/base.py` - Skill, SkillResult, SkillManifest, SkillRegistry
- `singularity/skill_loader.py` - Auto-discovers skills from directory
- `singularity/service_api.py` - FastAPI REST interface + messaging endpoints
- `singularity/skills/budget_planner.py` - NEW: Budget-aware goal planner
- `tests/test_budget_planner.py` - NEW: 18 tests
