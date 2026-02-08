"""
WisentBot Skills - Modular capabilities for autonomous agents.

Skills provide specific capabilities that agents can use to interact
with the world. Each skill has a manifest describing its actions.
"""

from .base import Skill, SkillRegistry, SkillManifest, SkillAction, SkillResult
from .browser import BrowserSkill
from .content import ContentCreationSkill
from .email import EmailSkill
from .filesystem import FilesystemSkill
from .github import GitHubSkill
from .mcp_client import MCPClientSkill
from .namecheap import NamecheapSkill
from .request import RequestSkill
from .shell import ShellSkill
from .twitter import TwitterSkill
from .vercel import VercelSkill
from .self_modify import SelfModifySkill
from .steering import SteeringSkill
from .memory import MemorySkill
from .orchestrator import OrchestratorSkill
from .crypto import CryptoSkill
from .experiment import ExperimentSkill
from .event import EventSkill
from .planner import PlannerSkill
from .outcome_tracker import OutcomeTracker
from .scheduler import SchedulerSkill
from .strategy import StrategySkill
from .goal_manager import GoalManagerSkill
from .task_delegator import TaskDelegator
from .knowledge_sharing import KnowledgeSharingSkill
from .revenue_services import RevenueServiceSkill
from .session_bootstrap import SessionBootstrapSkill
from .self_testing import SelfTestingSkill
from .cost_optimizer import CostOptimizerSkill
from .skill_marketplace_hub import SkillMarketplaceHub
from .self_tuning import SelfTuningSkill
from .self_assessment import SelfAssessmentSkill
from .cloudflare_dns import CloudflareDNSSkill
from .service_monitoring_dashboard import ServiceMonitoringDashboardSkill
from .tuning_presets import TuningPresetsSkill
from .fleet_health_manager import FleetHealthManagerSkill
from .revenue_catalog import RevenueServiceCatalogSkill
from .service_catalog import ServiceCatalogSkill
from .agent_checkpoint import AgentCheckpointSkill
from .workflow_analytics_bridge import WorkflowAnalyticsBridgeSkill
from .checkpoint_comparison import CheckpointComparisonAnalyticsSkill
from .revenue_analytics_dashboard import RevenueAnalyticsDashboardSkill

__all__ = [
    # Base
    "Skill",
    "SkillRegistry",
    "SkillManifest",
    "SkillAction",
    "SkillResult",
    # Skills
    "BrowserSkill",
    "ContentCreationSkill",
    "EmailSkill",
    "FilesystemSkill",
    "GitHubSkill",
    "MCPClientSkill",
    "NamecheapSkill",
    "RequestSkill",
    "ShellSkill",
    "TwitterSkill",
    "VercelSkill",
    "SelfModifySkill",
    "SteeringSkill",
    "MemorySkill",
    "OrchestratorSkill",
    "CryptoSkill",
    "ExperimentSkill",
    "EventSkill",
    "PlannerSkill",
    "OutcomeTracker",
    "SchedulerSkill",
    "StrategySkill",
    "GoalManagerSkill",
    "TaskDelegator",
    "KnowledgeSharingSkill",
    "RevenueServiceSkill",
    "SessionBootstrapSkill",
    "SelfTestingSkill",
    "CostOptimizerSkill",
    "SkillMarketplaceHub",
    "SelfTuningSkill",
    "SelfAssessmentSkill",
    "CloudflareDNSSkill",
    "ServiceMonitoringDashboardSkill",
    "TuningPresetsSkill",
    "FleetHealthManagerSkill",
    "RevenueServiceCatalogSkill",
    "ServiceCatalogSkill",
    "AgentCheckpointSkill",
    "WorkflowAnalyticsBridgeSkill",
    "CheckpointComparisonAnalyticsSkill",
    "RevenueAnalyticsDashboardSkill",
]
