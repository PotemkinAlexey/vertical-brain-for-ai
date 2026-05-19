"""Typed domain errors raised by the storage operation executor.

Every class subclasses ``ValueError``: the executor has always rejected bad
operations with ``ValueError``, callers (including the MCP server) and tests
catch it, and the human-readable messages are part of the agent-facing
contract. Subclassing keeps all of that working while letting callers branch
on the specific failure when they need to.
"""
from __future__ import annotations


class VerticalBrainError(ValueError):
    """Base class for Vertical Brain domain errors."""


class ValidationError(VerticalBrainError):
    """A StorageOperation failed schema or semantic validation before apply."""


class DuplicateBronzeError(VerticalBrainError):
    """An identical active Bronze chunk already exists at the target namespace."""


class SilverConflictError(VerticalBrainError):
    """append_chunk(layer=silver) hit an existing active Silver — use update_silver."""


class SilverUpdateError(VerticalBrainError):
    """update_silver was rejected: stale/mismatched current_silver_id or bad source."""


class GoldGroundingError(VerticalBrainError):
    """append_gold_aspect has no active Silver chunk to ground the aspect in."""


class ChunkNotFoundError(VerticalBrainError):
    """A referenced chunk id does not exist in storage."""


class ImmutableChunkError(VerticalBrainError):
    """An operation tried to modify an immutable chunk without force_immutable."""
