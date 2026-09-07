from typing import get_args

from bella_charm_agent.state import Step
from bella_charm_agent.step_config import STEP_CONFIG

# Handled by their own graph nodes, not the generic step-answer matcher --
# they have no fixed question to re-ask, so no STEP_CONFIG entry.
STEPS_WITHOUT_CONFIG = {"start", "order_placed"}


def test_every_step_has_a_config_entry():
    all_steps = set(get_args(Step))
    steps_needing_config = all_steps - STEPS_WITHOUT_CONFIG

    # Equality (not just "config covers the needed steps") also catches a
    # stray/misspelled key in STEP_CONFIG that doesn't match a real Step.
    assert set(STEP_CONFIG.keys()) == steps_needing_config
