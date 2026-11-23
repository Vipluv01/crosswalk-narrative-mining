"""STAMP safety constraint for an instrumented pedestrian crossing.

Constraint (from the brief): if crossing sensors flag an active pedestrian
while a vehicle's forward view is blocked, override the vehicle's speed
setting and enforce a mandatory 15 km/h limit.

The brief's rule only says what to do when both signals are healthy and
true. The governor below adds what the STPA in the report shows is needed
for the rule to be safe: fail-safe handling of missing feedback, a hold
time before release, and a bounded deceleration so the override does not
itself cause a rear-end collision.

Run:  .venv/bin/python crossing_governor.py   -> results/constraint.json
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

KMH = 1 / 3.6
CAP_KMH = 15.0


@dataclass(frozen=True)
class Config:
    cap_kmh: float = CAP_KMH
    hold_s: float = 3.0            # keep the cap this long after the hazard clears
    heartbeat_timeout_s: float = 0.5   # feedback older than this is treated as missing
    max_decel: float = 3.0         # m/s^2, service-brake ramp towards the cap


@dataclass(frozen=True)
class Inputs:
    t: float                               # s
    in_geofence: bool                      # vehicle inside the crossing approach zone
    pedestrian_active: Optional[bool]      # None = sensor fault / no data
    view_blocked: Optional[bool]           # None = perception cannot tell
    last_heartbeat_t: Optional[float]      # last valid roadside-unit message


@dataclass(frozen=True)
class Command:
    cap_kmh: Optional[float]   # None = no override, posted limit applies
    reason: str


class CrossingGovernor:
    """Deterministic on purpose: this is the constraint, not a model."""

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self._hazard_last_seen: Optional[float] = None

    def step(self, x: Inputs) -> Command:
        c = self.cfg
        if not x.in_geofence:
            self._hazard_last_seen = None
            return Command(None, "outside geofence")

        stale = x.last_heartbeat_t is None or (x.t - x.last_heartbeat_t) > c.heartbeat_timeout_s
        # Missing feedback is never read as "no hazard" (UCA-1).
        ped = True if (stale or x.pedestrian_active is None) else x.pedestrian_active
        blocked = True if x.view_blocked is None else x.view_blocked

        if ped and blocked:
            self._hazard_last_seen = x.t
            if stale:
                return Command(c.cap_kmh, "fail-safe: roadside link lost")
            if x.pedestrian_active is None:
                return Command(c.cap_kmh, "fail-safe: pedestrian sensor fault")
            if x.view_blocked is None:
                return Command(c.cap_kmh, "fail-safe: view state unknown")
            return Command(c.cap_kmh, "pedestrian active and view blocked")

        if self._hazard_last_seen is not None and (x.t - self._hazard_last_seen) < c.hold_s:
            return Command(c.cap_kmh, "hold after hazard cleared")   # UCA-4
        self._hazard_last_seen = None
        return Command(None, "no hazard")

    def target_speed(self, v_kmh: float, cmd: Command, dt: float) -> float:
        """Speed the vehicle may hold after dt: ramps down, never snaps (UCA-2)."""
        if cmd.cap_kmh is None or v_kmh <= cmd.cap_kmh:
            return v_kmh
        return max(cmd.cap_kmh, v_kmh - self.cfg.max_decel * dt / KMH)


# ---- kinematics used to judge the 15 km/h figure ---------------------------

def stopping_distance(v_kmh: float, reaction_s: float, decel: float) -> float:
    v = v_kmh * KMH
    return v * reaction_s + v * v / (2 * decel)


def impact_speed(v_kmh: float, sight_m: float, reaction_s: float, decel: float) -> float:
    """Speed (km/h) at which a pedestrian first visible sight_m ahead is reached."""
    v = v_kmh * KMH
    brake_room = sight_m - v * reaction_s
    if brake_room <= 0:
        return v_kmh
    return math.sqrt(max(0.0, v * v - 2 * decel * brake_room)) / KMH


def approach_zone(v_from_kmh: float, cfg: Config, latency_s: float) -> float:
    """Geofence length needed to get from v_from to the cap at the comfort ramp."""
    v0, v1 = v_from_kmh * KMH, cfg.cap_kmh * KMH
    return v0 * latency_s + max(0.0, v0 * v0 - v1 * v1) / (2 * cfg.max_decel)


def main() -> None:
    reaction, decel, latency = 1.0, 6.5, 0.5   # assumptions, stated in the report
    speeds = {"15 km/h (cap)": 15.0, "20 mph": 32.2, "30 mph": 48.3, "40 mph": 64.4}
    sights = [3, 5, 8, 12, 20, 30]
    out = {
        "assumptions": {"reaction_s": reaction, "emergency_decel_ms2": decel,
                        "v2i_latency_s": latency, "ramp_decel_ms2": Config().max_decel},
        "stopping_distance_m": {k: stopping_distance(v, reaction, decel) for k, v in speeds.items()},
        "impact_speed_kmh": {k: {str(s): impact_speed(v, s, reaction, decel) for s in sights}
                             for k, v in speeds.items()},
        "approach_zone_m": {k: approach_zone(v, Config(), latency) for k, v in speeds.items() if v > CAP_KMH},
    }
    p = Path(__file__).parent / "results"
    p.mkdir(exist_ok=True)
    (p / "constraint.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
