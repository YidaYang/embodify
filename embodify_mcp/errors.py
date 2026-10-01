"""Custom exceptions."""

from __future__ import annotations


class EmbodifyError(Exception):
    """Base class of all Embodify exceptions."""


class SimNotAvailableError(EmbodifyError):
    """The real simulator (libero / robosuite / mujoco) is not installed.

    This is not a bug: pure-logic tests and local development do not need it; only running a
    real episode does.
    """
