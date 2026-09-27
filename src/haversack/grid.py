"""Axis-aligned sampling grids in a canonical frame: labelfield's, re-exported.

The kernel layer moved to labelfield (github.com/mhalle/labelfield) on 2026-09-27; this module
keeps `haversack.grid` and `haversack.Grid` working.
"""
from __future__ import annotations

from labelfield.grid import Grid, Shape3, Vec3, _shape3, _vec3

__all__ = ["Grid", "Shape3", "Vec3"]
