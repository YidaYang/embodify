"""Calibration values for the stepped OSC pose controller.

Verify against the installed robot/controller before setting the provenance
flag. That flag records an operator declaration, not an automatic safety gate.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class ControllerParams:
    max_position_delta_m: float = 0.05
    max_rotation_delta_rad: float = 0.5
    gripper_close_sign: float = 1.0
    verified_against_facts: bool = False
