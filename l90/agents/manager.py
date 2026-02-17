"""Manager agent — central orchestrator. Never directly answers the user."""

from __future__ import annotations

import logging
from typing import Any

from l90.agents.base import BaseAgent
from l90.agents.planner import PlannerLayer
from l90.blackboard.blackboard import Blackboard
from l90.modes.enforcement import ModeEnforcer
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)


class ManagerAgent(BaseAgent):
    """Central orchestrator for the L90 pipeline.

    Responsibilities:
    1. Validates the incoming query and mode
    2. Delegates planning to PlannerLayer
    3. Sets up allowed sources via ModeEnforcer
    4. Writes the execution plan to the Blackboard
    5. NEVER directly answers the user

    The LangGraph graph reads the Blackboard state set by the Manager
    to route to the correct downstream nodes.
    """

    def __init__(self, trace_logger: ReasoningTraceLogger | None = None) -> None:
        super().__init__(name="ManagerAgent", trace_logger=trace_logger)
        self._planner = PlannerLayer(trace_logger=trace_logger)
        self._mode_enforcer = ModeEnforcer()

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Analyze query, determine plan, set up the Blackboard for downstream agents."""

        self._log_trace(
            phase="orchestration",
            action="manager_start",
            input_summary=f"Query: {blackboard.query[:100]}",
        )

        # Step 1: Enforce mode-based access control
        allowed = self._mode_enforcer.get_allowed_collections(
            mode=blackboard.mode,
            user_id=blackboard.user_id,
            workspace_id=blackboard.workspace_id,
        )
        blackboard.allowed_sources = allowed

        self._log_trace(
            phase="orchestration",
            action="mode_enforcement",
            decision=f"Mode={blackboard.mode}, Allowed={allowed}",
        )

        # Step 2: Delegate to PlannerLayer for strategy generation
        plan = await self._planner.create_plan(blackboard)

        # Step 3: Override allowed collections from plan (must still respect mode)
        # The plan may suggest collections but mode enforcer has final say
        plan_collections = set(plan.allowed_collections)
        enforced_collections = set(allowed)
        final_collections = list(plan_collections & enforced_collections)

        if not final_collections:
            # Plan requested collections outside mode boundary — use enforced
            final_collections = allowed
            logger.warning(
                "Plan collections %s outside mode boundary. Using enforced: %s",
                plan.allowed_collections,
                allowed,
            )

        blackboard.allowed_sources = final_collections

        self._log_trace(
            phase="orchestration",
            action="manager_complete",
            output_summary=(
                f"Plan: strategy={plan.retrieval_strategy}, "
                f"agents={plan.num_retrieval_agents}, "
                f"collections={final_collections}"
            ),
            decision="Blackboard prepared for retrieval swarm",
        )

        return blackboard
