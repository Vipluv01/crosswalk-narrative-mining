import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from crossing_governor import (Config, CrossingGovernor, Inputs,  # noqa: E402
                               approach_zone, impact_speed, stopping_distance)


def inp(t=0.0, geo=True, ped=True, blocked=True, hb=0.0):
    return Inputs(t, geo, ped, blocked, hb)


def test_brief_rule_caps_at_15():
    assert CrossingGovernor().step(inp()).cap_kmh == 15.0


@pytest.mark.parametrize("ped,blocked", [(False, True), (True, False), (False, False)])
def test_no_override_unless_both_true(ped, blocked):
    assert CrossingGovernor().step(inp(ped=ped, blocked=blocked)).cap_kmh is None


def test_outside_geofence_never_overrides():
    assert CrossingGovernor().step(inp(geo=False)).cap_kmh is None


def test_pedestrian_sensor_fault_is_fail_safe():
    cmd = CrossingGovernor().step(inp(ped=None))
    assert cmd.cap_kmh == 15.0 and "fault" in cmd.reason


def test_unknown_view_state_is_fail_safe():
    assert CrossingGovernor().step(inp(blocked=None)).cap_kmh == 15.0


def test_lost_link_is_fail_safe_even_if_last_message_said_clear():
    cmd = CrossingGovernor().step(inp(t=2.0, ped=False, hb=1.0))
    assert cmd.cap_kmh == 15.0 and "link" in cmd.reason


def test_lost_link_with_clear_view_does_not_override():
    # the driver / perception can see the crossing, so the cap adds nothing
    assert CrossingGovernor().step(inp(t=2.0, ped=False, blocked=False, hb=1.0)).cap_kmh is None


def test_cap_held_then_released():
    g = CrossingGovernor(Config(hold_s=3.0))
    g.step(inp(t=0.0))
    assert g.step(inp(t=2.9, ped=False, hb=2.9)).cap_kmh == 15.0
    assert g.step(inp(t=3.1, ped=False, hb=3.1)).cap_kmh is None


def test_speed_ramps_and_never_undershoots_cap():
    g = CrossingGovernor()
    cmd = g.step(inp())
    v = 48.3
    first = g.target_speed(v, cmd, 0.1)
    assert 15.0 < first < v
    assert (v - first) / 3.6 / 0.1 == pytest.approx(g.cfg.max_decel)
    for _ in range(200):
        v = g.target_speed(v, cmd, 0.1)
    assert v == 15.0
    assert g.target_speed(10.0, cmd, 0.1) == 10.0   # never speeds a vehicle up


def test_kinematics():
    assert stopping_distance(0, 1.0, 6.5) == 0
    assert impact_speed(15, 100, 1.0, 6.5) == 0
    assert impact_speed(48.3, 5, 1.0, 6.5) == pytest.approx(48.3)   # still reacting
    assert impact_speed(48.3, stopping_distance(48.3, 1.0, 6.5), 1.0, 6.5) == pytest.approx(0, abs=1e-3)
    assert approach_zone(15, Config(), 0.5) == pytest.approx(15 / 3.6 * 0.5)
