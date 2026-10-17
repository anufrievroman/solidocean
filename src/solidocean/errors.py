"""Exceptions shared across the core, below any physics (meshing raises Cancelled too)."""

from __future__ import annotations


class Cancelled(Exception):
    """Work stopped by the user (the cancel event was set)."""
