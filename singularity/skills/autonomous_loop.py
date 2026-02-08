#!/usr/bin/env python3
"""
AutonomousLoopSkill - The central executive for fully autonomous agent operation.

This is the "brain stem" that connects all other skills into a continuous
autonomous decision-execute-learn cycle. Without this, the agent needs a human
to tell it what to do. With it, the agent can:

1. ASSESS - Survey current state across all pillars (via strategy/goal_manager)
2. DECIDE - Pick the highest-priority task to work on next
3. PLAN   - Break the chosen task into executable steps
4. ACT    - Execute steps via skill orchestration
5. MEASURE - Record outcomes (via outcome_tracker/performance)
6. LEARN  - Adapt behavior based on results (via feedback_loop)
7. REPEAT - Loop back to ASSESS

The loop can run continuously (daemon mode), step-by-step (manual advance),
or for N iterations (bounded mode). It maintains a persistent execution journal
so the agent can review its autonomous decision history.

Pillars served: Goal Setting (primary), Self-Improvement (feedback integration)
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Skill, SkillResult, SkillManifest, SkillAction
from .circuit_breaker import wire_adaptive_thresholds


LOOP_STATE_FILE = Path(__file__).parent.parent / "data" / "autonomous_loop.json"


class LoopPhase:
    """Phases of the autonomous loop."""
    IDLE = "idle"
    ASSESS = "assess"
    DECIDE = "decide"
    PLAN = "plan"
    ACT = "act"
    MEASURE = "measure"
    LEARN = "learn"


class AutonomousLoopSkill(Skill):
    """
    Central executive for autonomous agent operation.

    Orchestrates the full assess-decide-plan-act-measure-learn cycle
    by composing existing skills (strategy, goal_manager, feedback_loop,
    outcome_tracker, session_bootstrap) into a coherent autonomous loop.
    """

    def __init__(self, credentials: Dict[str, str] = None):
        super().__init__(credentials)
        self._ensure_data()

    def _ensure_data(self):
        LOOP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not LOOP_STATE_FILE.exists():
            self._save(self._default_state())

    def _default_state(self) -> Dict:
        return {
            "current_phase": LoopPhase.IDLE,
            "iteration_count": 0,
            "current_task": None,
            "current_plan": None,
            "journal": [],  # Full history of loop iterations
            "config": {
                "max_iterations": 0,       # 0 = unlimited
                "pause_between_iterations": 0,  # seconds
                "auto_learn": True,         # Run feedback loop after each iteration
                "skip_assess_if_recent": 300,  # Skip assess if done in last N seconds
                "max_journal_entries": 200,
                "circuit_breaker_enabled": True,  # Check circuit breaker before skill execution
                "circuit_breaker_skip_self": True,  # Don't circuit-break the loop's own skill calls
            },
            "stats": {
                "total_iterations": 0,
                "successful_actions": 0,
                "failed_actions": 0,
                "total_revenue": 0.0,
                "total_cost": 0.0,
                "started_at": None,
                "last_iteration_at": None,
                "circuit_breaker_denials": 0,
                "circuit_breaker_recordings": 0,
            },
            "last_assessment": None,
            "last_assessment_at": None,
            "created_at": datetime.now().isoformat(),
        }

    def _load(self) -> Dict:
        try:
            with open(LOOP_STATE_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._default_state()

    def _save(self, state: Dict):
        LOOP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state["last_updated"] = datetime.now().isoformat()
        with open(LOOP_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            skill_id="autonomous_loop",
            name="Autonomous Loop",
            version="1.0.0",
            category="autonomy",
            description="Central executive for fully autonomous assess-decide-plan-act-measure-learn cycles",
            actions=[
                SkillAction(
                    name="step",
                    description="Execute one full iteration of the autonomous loop (assess->decide->plan->act->measure->learn)",
                    parameters={
                        "force_assess": {
                            "type": "boolean",
                            "required": False,
                            "description": "Force fresh assessment even if recent one exists"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="run",
                    description="Run the autonomous loop for N iterations (or until stopped)",
                    parameters={
                        "iterations": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of iterations (default: 1, 0 = unlimited)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="assess",
                    description="Run only the assessment phase - survey current state across all pillars",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="decide",
                    description="Run only the decision phase - pick the highest-priority task",
                    parameters={
                        "assessment": {
                            "type": "object",
                            "required": False,
                            "description": "Assessment data to use (uses cached if not provided)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="status",
                    description="Get the current loop status, phase, and statistics",
                    parameters={},
                    estimated_cost=0,
                ),
                SkillAction(
                    name="journal",
                    description="View the autonomous decision journal (history of loop iterations)",
                    parameters={
                        "limit": {
                            "type": "integer",
                            "required": False,
                            "description": "Max entries to return (default 10)"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="configure",
                    description="Update loop configuration (max_iterations, auto_learn, etc.)",
                    parameters={
                        "max_iterations": {
                            "type": "integer",
                            "required": False,
                            "description": "Max iterations per run (0=unlimited)"
                        },
                        "auto_learn": {
                            "type": "boolean",
                            "required": False,
                            "description": "Run feedback loop after each iteration"
                        },
                        "pause_between_iterations": {
                            "type": "number",
                            "required": False,
                            "description": "Seconds to pause between iterations"
                        },
                    },
                    estimated_cost=0,
                ),
                SkillAction(
                    name="reset",
                    description="Reset the loop state (keeps journal, clears current task/plan)",
                    parameters={},
                    estimated_cost=0,
                ),
            ],
            required_credentials=[],
        )

    def check_credentials(self) -> bool:
        return True

    async def execute(self, action: str, params: Dict) -> SkillResult:
        handlers = {
            "step": self._step,
            "run": self._run,
            "assess": self._assess,
            "decide": self._decide,
            "status": self._status,
            "journal": self._journal,
            "configure": self._configure,
            "reset": self._reset,
        }
        handler = handlers.get(action)
        if not handler:
            return SkillResult(success=False, message=f"Unknown action: {action}")
        return await handler(params)

    # ========== Core Loop ==========

    def _wire_adaptive_circuit_breaker(self):
        """Wire AdaptiveCircuitThresholdsSkill into CircuitBreakerSkill.

        Called at the start of each loop iteration to ensure per-skill
        adaptive thresholds are used when evaluating circuits.
        Fail-silent: does nothing if either skill is unavailable.
        """
        if not self.context:
            return
        try:
            wire_adaptive_thresholds(self.context._registry)
        except Exception:
            pass  # Fail silently - wiring is optional enhancement

    async def _step(self, params: Dict) -> SkillResult:
        """Execute one full iteration of the autonomous loop."""
        # Auto-wire adaptive thresholds into circuit breaker if both exist
        self._wire_adaptive_circuit_breaker()

        state = self._load()
        force_assess = params.get("force_assess", False)
        iteration_id = f"iter_{uuid.uuid4().hex[:8]}"
        iteration_start = time.time()

        journal_entry = {
            "id": iteration_id,
            "started_at": datetime.now().isoformat(),
            "phases": {},
            "outcome": None,
        }

        # Phase 1: ASSESS
        state["current_phase"] = LoopPhase.ASSESS
        self._save(state)

        assessment = await self._run_assessment(state, force_assess)
        journal_entry["phases"]["assess"] = {
            "result": "completed",
            "summary": assessment.get("summary", ""),
            "weakest_pillar": assessment.get("weakest_pillar", "unknown"),
        }

        # Phase 2: DECIDE
        state["current_phase"] = LoopPhase.DECIDE
        self._save(state)

        decision = await self._run_decision(assessment)
        journal_entry["phases"]["decide"] = {
            "result": "completed",
            "chosen_task": decision.get("task_description", ""),
            "chosen_pillar": decision.get("pillar", ""),
            "reasoning": decision.get("reasoning", ""),
        }

        if not decision.get("task_description"):
            journal_entry["outcome"] = "no_task_found"
            state["current_phase"] = LoopPhase.IDLE
            self._append_journal(state, journal_entry)
            self._save(state)
            return SkillResult(
                success=True,
                message="Assessment complete but no actionable task found. All pillars may be well-served.",
                data={"iteration_id": iteration_id, "assessment": assessment}
            )

        # Phase 3: PLAN
        state["current_phase"] = LoopPhase.PLAN
        state["current_task"] = decision
        self._save(state)

        plan = await self._run_planning(decision)
        journal_entry["phases"]["plan"] = {
            "result": "completed",
            "steps_count": len(plan.get("steps", [])),
        }
        state["current_plan"] = plan

        # Phase 4: ACT
        state["current_phase"] = LoopPhase.ACT
        self._save(state)

        action_results = await self._run_actions(plan)
        journal_entry["phases"]["act"] = {
            "result": "completed" if action_results.get("success") else "partial",
            "steps_executed": action_results.get("steps_executed", 0),
            "steps_succeeded": action_results.get("steps_succeeded", 0),
        }

        # Sync circuit breaker events after skill executions
        await self._sync_cb_event_bridge(state)

        # Phase 5: MEASURE
        state["current_phase"] = LoopPhase.MEASURE
        self._save(state)

        measurement = await self._run_measurement(decision, action_results)
        journal_entry["phases"]["measure"] = {
            "result": "completed",
            "success": measurement.get("success", False),
            "revenue": measurement.get("revenue", 0),
            "cost": measurement.get("cost", 0),
        }

        # Phase 6: LEARN
        config = state.get("config", {})
        if config.get("auto_learn", True):
            state["current_phase"] = LoopPhase.LEARN
            self._save(state)

            learning = await self._run_learning(measurement)
            journal_entry["phases"]["learn"] = {
                "result": "completed",
                "adaptations": learning.get("adaptations_count", 0),
            }

        # Finalize
        duration = time.time() - iteration_start
        overall_success = action_results.get("success", False)

        journal_entry["outcome"] = "success" if overall_success else "partial"
        journal_entry["duration_seconds"] = round(duration, 2)
        journal_entry["completed_at"] = datetime.now().isoformat()

        # Update stats
        stats = state.get("stats", {})
        stats["total_iterations"] = stats.get("total_iterations", 0) + 1
        if overall_success:
            stats["successful_actions"] = stats.get("successful_actions", 0) + 1
        else:
            stats["failed_actions"] = stats.get("failed_actions", 0) + 1
        stats["total_revenue"] = stats.get("total_revenue", 0) + measurement.get("revenue", 0)
        stats["total_cost"] = stats.get("total_cost", 0) + measurement.get("cost", 0)
        stats["last_iteration_at"] = datetime.now().isoformat()
        if not stats.get("started_at"):
            stats["started_at"] = datetime.now().isoformat()
        state["stats"] = stats

        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["current_phase"] = LoopPhase.IDLE
        state["current_task"] = None
        state["current_plan"] = None
        self._append_journal(state, journal_entry)
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Loop iteration {iteration_id} completed: {journal_entry['outcome']} "
                    f"(task: {decision.get('task_description', 'N/A')[:60]}, "
                    f"duration: {duration:.1f}s)",
            data={
                "iteration_id": iteration_id,
                "outcome": journal_entry["outcome"],
                "task": decision.get("task_description", ""),
                "pillar": decision.get("pillar", ""),
                "duration_seconds": round(duration, 2),
                "phases": journal_entry["phases"],
                "stats": stats,
            }
        )

    async def _run(self, params: Dict) -> SkillResult:
        """Run the loop for N iterations."""
        iterations = params.get("iterations", 1)
        if iterations < 0:
            return SkillResult(success=False, message="iterations must be >= 0")

        results = []
        max_iter = iterations if iterations > 0 else 100  # Safety cap

        for i in range(max_iter):
            result = await self._step({"force_assess": (i == 0)})
            results.append({
                "iteration": i + 1,
                "success": result.success,
                "message": result.message,
                "outcome": result.data.get("outcome", ""),
            })

            # Check if we should stop (no task found = nothing to do)
            if result.data.get("outcome") == "no_task_found":
                break

            # Pause between iterations if configured
            state = self._load()
            pause = state.get("config", {}).get("pause_between_iterations", 0)
            if pause > 0 and i < max_iter - 1:
                import asyncio
                await asyncio.sleep(pause)

        completed = len(results)
        successful = sum(1 for r in results if r.get("outcome") == "success")

        return SkillResult(
            success=True,
            message=f"Autonomous loop completed {completed} iteration(s): "
                    f"{successful} successful, {completed - successful} other",
            data={
                "iterations_completed": completed,
                "iterations_successful": successful,
                "results": results,
            }
        )

    # ========== Individual Phases ==========

    async def _assess(self, params: Dict) -> SkillResult:
        """Run assessment phase only."""
        state = self._load()
        assessment = await self._run_assessment(state, force=True)
        state["last_assessment"] = assessment
        state["last_assessment_at"] = datetime.now().isoformat()
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Assessment complete. Weakest pillar: {assessment.get('weakest_pillar', 'unknown')}. "
                    f"Summary: {assessment.get('summary', 'N/A')[:100]}",
            data=assessment,
        )

    async def _decide(self, params: Dict) -> SkillResult:
        """Run decision phase only."""
        assessment = params.get("assessment")
        if not assessment:
            state = self._load()
            assessment = state.get("last_assessment") or {}
            if not assessment:
                return SkillResult(
                    success=False,
                    message="No assessment available. Run 'assess' first."
                )

        decision = await self._run_decision(assessment)
        if not decision.get("task_description"):
            return SkillResult(
                success=True,
                message="No actionable task identified from current assessment.",
                data=decision,
            )

        return SkillResult(
            success=True,
            message=f"Decided: [{decision.get('pillar', '?')}] {decision.get('task_description', 'N/A')[:80]}",
            data=decision,
        )

    # ========== Phase Implementations ==========

    async def _run_assessment(self, state: Dict, force: bool = False) -> Dict:
        """
        Assess current state by querying strategy and goal_manager.
        Returns a dict with pillar scores, weakest pillar, and summary.
        """
        config = state.get("config", {})
        skip_threshold = config.get("skip_assess_if_recent", 300)

        # Use cached assessment if recent enough
        if not force and state.get("last_assessment_at"):
            try:
                last_time = datetime.fromisoformat(state["last_assessment_at"])
                age = (datetime.now() - last_time).total_seconds()
                if age < skip_threshold and state.get("last_assessment"):
                    return state["last_assessment"]
            except (ValueError, TypeError):
                pass

        assessment = {
            "pillars": {},
            "weakest_pillar": None,
            "strongest_pillar": None,
            "summary": "",
            "assessed_at": datetime.now().isoformat(),
        }

        # Try to get pillar assessment from strategy skill
        if self.context:
            strategy_result = await self.context.call_skill(
                "strategy", "assess", {}
            )
            if strategy_result.success and strategy_result.data:
                pillars_data = strategy_result.data.get("pillars", {})
                for pillar_id, pillar_info in pillars_data.items():
                    assessment["pillars"][pillar_id] = {
                        "score": pillar_info.get("score", 0),
                        "capabilities": pillar_info.get("capabilities", []),
                        "gaps": pillar_info.get("gaps", []),
                    }

                assessment["weakest_pillar"] = strategy_result.data.get(
                    "weakest_pillar", strategy_result.data.get("recommended_focus", "")
                )
                assessment["strongest_pillar"] = strategy_result.data.get("strongest_pillar", "")
                assessment["summary"] = strategy_result.data.get("summary", strategy_result.message)

            # Also check active goals
            goals_result = await self.context.call_skill(
                "goal_manager", "next", {}
            )
            if goals_result.success and goals_result.data:
                assessment["next_goal"] = {
                    "id": goals_result.data.get("goal_id", ""),
                    "title": goals_result.data.get("title", ""),
                    "pillar": goals_result.data.get("pillar", ""),
                    "priority": goals_result.data.get("priority", ""),
                }

        # If no strategy skill available, create basic assessment
        if not assessment["weakest_pillar"]:
            assessment["weakest_pillar"] = "goal_setting"
            assessment["summary"] = "No strategy skill available. Defaulting to goal_setting focus."

        # Cache the assessment
        state["last_assessment"] = assessment
        state["last_assessment_at"] = datetime.now().isoformat()

        return assessment

    async def _run_decision(self, assessment: Dict) -> Dict:
        """
        Decide what to work on based on assessment.
        Priority cascade:
        1. Active goal from goal_manager (if urgent/high priority)
        2. Weakest pillar gaps
        3. Revenue opportunities (if low balance)
        4. General self-improvement
        """
        decision = {
            "task_description": "",
            "pillar": "",
            "reasoning": "",
            "source": "",
            "skill_to_use": None,
            "action_to_take": None,
            "params": {},
        }

        # Priority 1: Check for urgent active goals
        next_goal = assessment.get("next_goal", {})
        if next_goal and next_goal.get("title"):
            decision["task_description"] = next_goal["title"]
            decision["pillar"] = next_goal.get("pillar", "other")
            decision["reasoning"] = (
                f"Active goal '{next_goal['title']}' has priority "
                f"'{next_goal.get('priority', 'medium')}' and is the next recommended task."
            )
            decision["source"] = "goal_manager"
            decision["goal_id"] = next_goal.get("id", "")
            return decision

        # Priority 2: Address weakest pillar
        weakest = assessment.get("weakest_pillar", "")
        pillar_data = assessment.get("pillars", {}).get(weakest, {})
        gaps = pillar_data.get("gaps", [])

        if weakest and gaps:
            decision["task_description"] = f"Address gap in {weakest}: {gaps[0]}"
            decision["pillar"] = weakest
            decision["reasoning"] = (
                f"Pillar '{weakest}' is weakest (score: {pillar_data.get('score', 0)}) "
                f"with {len(gaps)} gap(s). Addressing top gap: {gaps[0]}"
            )
            decision["source"] = "strategy_assessment"
            return decision

        if weakest:
            decision["task_description"] = f"Improve {weakest} pillar capabilities"
            decision["pillar"] = weakest
            decision["reasoning"] = f"Pillar '{weakest}' identified as weakest. General improvement needed."
            decision["source"] = "strategy_assessment"
            return decision

        # Priority 3: No specific task found
        decision["reasoning"] = "No urgent goals or pillar gaps identified."
        return decision

    async def _run_planning(self, decision: Dict) -> Dict:
        """
        Create an execution plan for the decided task.
        Uses task_queue or planner if available, otherwise creates a simple plan.
        """
        plan = {
            "task": decision.get("task_description", ""),
            "pillar": decision.get("pillar", ""),
            "steps": [],
            "created_at": datetime.now().isoformat(),
        }

        # If there's a specific skill/action already identified, use that
        if decision.get("skill_to_use") and decision.get("action_to_take"):
            plan["steps"].append({
                "step": 1,
                "description": decision["task_description"],
                "skill_id": decision["skill_to_use"],
                "action": decision["action_to_take"],
                "params": decision.get("params", {}),
                "status": "pending",
            })
            return plan

        # If goal_manager has milestones, use those
        if decision.get("source") == "goal_manager" and decision.get("goal_id") and self.context:
            goal_result = await self.context.call_skill(
                "goal_manager", "get", {"goal_id": decision["goal_id"]}
            )
            if goal_result.success and goal_result.data:
                milestones = goal_result.data.get("milestones", [])
                pending = [m for m in milestones if m.get("status") != "completed"]
                for i, milestone in enumerate(pending[:5], 1):
                    plan["steps"].append({
                        "step": i,
                        "description": milestone.get("title", milestone.get("description", "")),
                        "milestone_id": milestone.get("id", ""),
                        "skill_id": None,
                        "action": None,
                        "params": {},
                        "status": "pending",
                    })

        # Default: single-step plan
        if not plan["steps"]:
            plan["steps"].append({
                "step": 1,
                "description": decision.get("task_description", "Execute task"),
                "skill_id": None,
                "action": None,
                "params": {},
                "status": "pending",
            })

        return plan

    async def _check_circuit_breaker(self, skill_id: str, state: Dict) -> Dict:
        """Check if a skill is allowed to execute via circuit breaker.

        Returns dict with 'allowed' (bool) and 'reason' (str).
        If circuit_breaker is disabled or unavailable, always allows execution.
        """
        config = state.get("config", {})
        if not config.get("circuit_breaker_enabled", True):
            return {"allowed": True, "reason": "circuit_breaker_disabled"}

        # Don't circuit-break the loop's own internal calls
        skip_self = config.get("circuit_breaker_skip_self", True)
        internal_skills = {"autonomous_loop", "circuit_breaker", "outcome_tracker", "feedback_loop"}
        if skip_self and skill_id in internal_skills:
            return {"allowed": True, "reason": "internal_skill_exempt"}

        if not self.context:
            return {"allowed": True, "reason": "no_context"}

        try:
            check_result = await self.context.call_skill(
                "circuit_breaker", "check", {"skill_id": skill_id}
            )
            if check_result.success and check_result.data:
                allowed = check_result.data.get("allowed", True)
                reason = check_result.data.get("reason", "")
                return {"allowed": allowed, "reason": reason, "data": check_result.data}
            # If circuit_breaker skill returns failure (e.g., not registered), allow execution
            return {"allowed": True, "reason": "circuit_breaker_unavailable"}
        except Exception:
            # Circuit breaker itself failing should never block execution
            return {"allowed": True, "reason": "circuit_breaker_error"}

    async def _record_circuit_outcome(self, skill_id: str, success: bool,
                                       cost: float = 0.0, duration_ms: float = 0.0,
                                       error: str = "", state: Dict = None) -> None:
        """Record a skill execution outcome in the circuit breaker.

        Silently fails if circuit_breaker is unavailable or disabled.
        """
        config = (state or {}).get("config", {})
        if not config.get("circuit_breaker_enabled", True):
            return

        # Don't record internal skill calls
        skip_self = config.get("circuit_breaker_skip_self", True)
        internal_skills = {"autonomous_loop", "circuit_breaker", "outcome_tracker", "feedback_loop"}
        if skip_self and skill_id in internal_skills:
            return

        if not self.context:
            return

        try:
            await self.context.call_skill(
                "circuit_breaker", "record", {
                    "skill_id": skill_id,
                    "success": success,
                    "cost": cost,
                    "duration_ms": duration_ms,
                    "error": error[:200] if error else "",
                }
            )
        except Exception:
            pass  # Never let circuit breaker recording break the main loop

    async def _run_actions(self, plan: Dict) -> Dict:
        """Execute the planned steps with circuit breaker protection.

        Before each skill execution:
        1. Checks circuit breaker - if the circuit is open, skips the skill
        2. Executes the skill if allowed
        3. Records the outcome (success/failure) back to the circuit breaker

        This prevents the autonomous loop from repeatedly calling broken skills,
        saving budget and enabling graceful degradation.
        """
        state = self._load()
        results = {
            "success": False,
            "steps_executed": 0,
            "steps_succeeded": 0,
            "steps_denied": 0,
            "step_results": [],
            "total_revenue": 0.0,
            "total_cost": 0.0,
        }

        for step in plan.get("steps", []):
            step_result = {
                "step": step.get("step", 0),
                "description": step.get("description", ""),
                "success": False,
                "message": "",
                "circuit_breaker": None,
            }

            skill_id = step.get("skill_id")
            action = step.get("action")

            if skill_id and action and self.context:
                # Phase 1: Check circuit breaker
                cb_check = await self._check_circuit_breaker(skill_id, state)
                step_result["circuit_breaker"] = cb_check.get("reason", "")

                if not cb_check.get("allowed", True):
                    # Circuit is open - skip this skill
                    step_result["message"] = (
                        f"DENIED by circuit breaker: {skill_id} "
                        f"({cb_check.get('reason', 'circuit_open')})"
                    )
                    step["status"] = "circuit_denied"
                    results["steps_denied"] += 1

                    # Update stats
                    stats = state.get("stats", {})
                    stats["circuit_breaker_denials"] = stats.get("circuit_breaker_denials", 0) + 1
                    state["stats"] = stats
                else:
                    # Phase 2: Execute the skill
                    start_time = time.time()
                    try:
                        result = await self.context.call_skill(
                            skill_id, action, step.get("params", {})
                        )
                        duration_ms = (time.time() - start_time) * 1000
                        step_result["success"] = result.success
                        step_result["message"] = result.message[:200]
                        step["status"] = "completed" if result.success else "failed"
                        results["total_revenue"] += result.revenue
                        results["total_cost"] += result.cost

                        # Phase 3: Record outcome to circuit breaker
                        await self._record_circuit_outcome(
                            skill_id=skill_id,
                            success=result.success,
                            cost=result.cost,
                            duration_ms=duration_ms,
                            error="" if result.success else result.message[:200],
                            state=state,
                        )
                        stats = state.get("stats", {})
                        stats["circuit_breaker_recordings"] = stats.get("circuit_breaker_recordings", 0) + 1
                        state["stats"] = stats

                    except Exception as e:
                        duration_ms = (time.time() - start_time) * 1000
                        step_result["message"] = f"Error: {str(e)[:150]}"
                        step["status"] = "failed"

                        # Record failure to circuit breaker
                        await self._record_circuit_outcome(
                            skill_id=skill_id,
                            success=False,
                            cost=0.0,
                            duration_ms=duration_ms,
                            error=str(e)[:200],
                            state=state,
                        )
            else:
                # No specific skill action - record as a recommendation
                step_result["success"] = True
                step_result["message"] = f"Recommended action: {step.get('description', '')}"
                step["status"] = "completed"

            results["steps_executed"] += 1
            if step_result["success"]:
                results["steps_succeeded"] += 1
            results["step_results"].append(step_result)

        results["success"] = results["steps_succeeded"] > 0
        return results

    async def _sync_cb_event_bridge(self, state: Dict):
        """Sync circuit breaker event bridge to emit events for state changes.
        
        Fail-silent: if the bridge skill isn't registered, just skip.
        """
        try:
            if self.context:
                await self.context.call_skill(
                    "circuit_breaker_event_bridge", "sync", {}
                )
                stats = state.get("stats", {})
                stats["cb_bridge_syncs"] = stats.get("cb_bridge_syncs", 0) + 1
                state["stats"] = stats
        except Exception:
            pass  # Bridge not registered or unavailable - that's OK

    async def _run_measurement(self, decision: Dict, action_results: Dict) -> Dict:
        """Record outcomes using outcome_tracker if available."""
        measurement = {
            "success": action_results.get("success", False),
            "steps_executed": action_results.get("steps_executed", 0),
            "steps_succeeded": action_results.get("steps_succeeded", 0),
            "revenue": action_results.get("total_revenue", 0),
            "cost": action_results.get("total_cost", 0),
            "measured_at": datetime.now().isoformat(),
        }

        # Try to log outcome via outcome_tracker
        if self.context:
            outcome_result = await self.context.call_skill(
                "outcome_tracker", "log", {
                    "action": f"autonomous_loop:{decision.get('pillar', 'unknown')}",
                    "skill_id": "autonomous_loop",
                    "success": measurement["success"],
                    "details": f"Task: {decision.get('task_description', 'N/A')[:100]}. "
                               f"Steps: {measurement['steps_succeeded']}/{measurement['steps_executed']}",
                }
            )
            if outcome_result.success:
                measurement["tracked"] = True

        return measurement

    async def _run_learning(self, measurement: Dict) -> Dict:
        """Run feedback loop to adapt based on results."""
        learning = {
            "adaptations_count": 0,
            "learned_at": datetime.now().isoformat(),
        }

        if self.context:
            feedback_result = await self.context.call_skill(
                "feedback_loop", "analyze", {}
            )
            if feedback_result.success and feedback_result.data:
                learning["adaptations_count"] = len(
                    feedback_result.data.get("adaptations", [])
                )
                learning["patterns_detected"] = feedback_result.data.get("patterns", [])

        return learning

    # ========== Info Actions ==========

    async def _status(self, params: Dict) -> SkillResult:
        """Get current loop status."""
        state = self._load()

        return SkillResult(
            success=True,
            message=f"Loop phase: {state.get('current_phase', 'idle')} | "
                    f"Iterations: {state.get('stats', {}).get('total_iterations', 0)} | "
                    f"Success rate: {self._success_rate(state):.0%}",
            data={
                "phase": state.get("current_phase", "idle"),
                "iteration_count": state.get("iteration_count", 0),
                "current_task": state.get("current_task"),
                "stats": state.get("stats", {}),
                "config": state.get("config", {}),
                "last_assessment_at": state.get("last_assessment_at"),
                "success_rate": self._success_rate(state),
            }
        )

    async def _journal(self, params: Dict) -> SkillResult:
        """View the autonomous decision journal."""
        state = self._load()
        limit = params.get("limit", 10)

        journal = state.get("journal", [])
        entries = journal[-limit:][::-1]  # Most recent first

        summaries = []
        for entry in entries:
            decide_phase = entry.get("phases", {}).get("decide", {})
            summaries.append({
                "id": entry.get("id", ""),
                "started_at": entry.get("started_at", ""),
                "outcome": entry.get("outcome", ""),
                "task": decide_phase.get("chosen_task", "N/A")[:80],
                "pillar": decide_phase.get("chosen_pillar", ""),
                "duration": entry.get("duration_seconds", 0),
            })

        return SkillResult(
            success=True,
            message=f"{len(summaries)} journal entries (of {len(journal)} total)",
            data={
                "entries": summaries,
                "total_entries": len(journal),
            }
        )

    async def _configure(self, params: Dict) -> SkillResult:
        """Update loop configuration."""
        state = self._load()
        config = state.get("config", {})

        updated = []
        for key in ["max_iterations", "auto_learn", "pause_between_iterations"]:
            if key in params:
                config[key] = params[key]
                updated.append(f"{key}={params[key]}")

        state["config"] = config
        self._save(state)

        return SkillResult(
            success=True,
            message=f"Configuration updated: {', '.join(updated) if updated else 'no changes'}",
            data={"config": config}
        )

    async def _reset(self, params: Dict) -> SkillResult:
        """Reset loop state, preserving journal and stats."""
        state = self._load()
        state["current_phase"] = LoopPhase.IDLE
        state["current_task"] = None
        state["current_plan"] = None
        state["iteration_count"] = 0
        self._save(state)

        return SkillResult(
            success=True,
            message="Loop state reset. Journal and stats preserved.",
            data={"phase": LoopPhase.IDLE}
        )

    # ========== Helpers ==========

    def _success_rate(self, state: Dict) -> float:
        """Calculate success rate from stats."""
        stats = state.get("stats", {})
        total = stats.get("successful_actions", 0) + stats.get("failed_actions", 0)
        if total == 0:
            return 0.0
        return stats.get("successful_actions", 0) / total

    def _append_journal(self, state: Dict, entry: Dict):
        """Append to journal, respecting max size."""
        journal = state.get("journal", [])
        journal.append(entry)
        max_entries = state.get("config", {}).get("max_journal_entries", 200)
        if len(journal) > max_entries:
            journal = journal[-max_entries:]
        state["journal"] = journal
