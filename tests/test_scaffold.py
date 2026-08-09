from pathlib import Path

from contra_rl.envs.actions import ACTION_SETS
from contra_rl.envs.contra_env import ContraEnvError, validate_rom_path
from contra_rl.envs.rewards import RewardParts


def test_action_sets_present():
    assert "SIMPLE_MOVEMENT" in ACTION_SETS
    assert len(ACTION_SETS["SIMPLE_MOVEMENT"]) > 0


def test_reward_parts_total():
    parts = RewardParts(progress=1.0, death=-2.0, time=-0.5)
    assert parts.total == -1.5


def test_validate_rom_path_missing():
    missing_rom = Path("roms/definitely-missing-test-rom.nes")
    try:
        validate_rom_path(missing_rom)
    except ContraEnvError as exc:
        assert "ROM not found" in str(exc)
    else:
        raise AssertionError("expected missing ROM to raise ContraEnvError")
