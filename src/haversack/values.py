"""Small value types haversack needs, free of any framework.

These mirror the MLX toolkit's ``Geometry`` and ``LabelSchema`` deliberately: that package
imports ``mlx.core`` at module level, which does not exist off Apple silicon, so depending on
it would make haversack unimportable on exactly the machines the torch path exists for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Geometry:
    """Voxel-grid placement in physical space, SimpleITK's conventions.

    ``spacing_zyx`` / ``shape_zyx`` are in array order; ``origin_xyz`` and the 9-tuple
    row-major ``direction_xyz`` are in SimpleITK's X, Y, Z order.
    """

    spacing_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction_xyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "spacing_zyx", tuple(float(s) for s in self.spacing_zyx))
        object.__setattr__(self, "shape_zyx", tuple(int(s) for s in self.shape_zyx))
        object.__setattr__(self, "origin_xyz", tuple(float(o) for o in self.origin_xyz))
        object.__setattr__(self, "direction_xyz", tuple(float(d) for d in self.direction_xyz))
        if len(self.direction_xyz) != 9:
            raise ValueError(f"direction_xyz must have 9 entries; got {len(self.direction_xyz)}")

    #: This type keeps SimpleITK's split - array-order sizes beside world-order cosines -
    #: because that is what the readers and writers around it speak. duckn, NRRD and
    #: rankfield (from 0.3) keep ONE order instead: a direction VECTOR per array axis,
    #: its length the spacing. The two records below are the only place the two forms
    #: meet, so the reversal and transpose are written once rather than at each boundary.

    def to_record(self) -> dict:
        """``{shape, directions, origin}`` - duckn's form, which is NRRD's."""
        d = self.direction_xyz
        cols = [(d[0], d[3], d[6]), (d[1], d[4], d[7]), (d[2], d[5], d[8])]   # x, y, z
        rows = list(reversed(cols))                                           # z, y, x
        return {"shape": [int(v) for v in self.shape_zyx],
                "directions": [[float(c * sp) for c in row]
                               for row, sp in zip(rows, self.spacing_zyx)],
                "origin": [float(v) for v in self.origin_xyz]}

    @classmethod
    def from_record(cls, rec) -> "Geometry":
        """The inverse. Takes the mapping :meth:`to_record` writes, or any object with
        ``shape`` / ``directions`` / ``origin`` - a ``rankfield.Geometry`` is one."""
        import math
        get = rec.get if isinstance(rec, Mapping) else (lambda k, _d=None: getattr(rec, k, _d))
        missing = [k for k in ("shape", "directions", "origin") if get(k) is None]
        if missing:
            keys = sorted(rec) if isinstance(rec, Mapping) else missing
            raise ValueError(f"a geometry record needs shape, directions and origin; got {keys}"
                             " (a record with spacing_zyx / direction_xyz predates rankfield 0.3)")
        rows = [[float(v) for v in row] for row in get("directions")]          # z, y, x
        spacing = [math.sqrt(sum(v * v for v in row)) for row in rows]
        cos = [[v / sp if sp else 0.0 for v in row] for row, sp in zip(rows, spacing)]
        cx, cy, cz = cos[2], cos[1], cos[0]                                    # columns x, y, z
        return cls(spacing_zyx=tuple(spacing), shape_zyx=tuple(int(v) for v in get("shape")),
                   origin_xyz=tuple(float(v) for v in get("origin")),
                   direction_xyz=tuple(float(v) for r in range(3) for v in (cx[r], cy[r], cz[r])))


@dataclass(frozen=True)
class LabelSchema:
    """Integer label -> name. Region (sigmoid-head) models additionally carry a paint order."""

    names: Mapping[int, str]
    paint_priority: tuple[int, ...] = ()

    @property
    def is_region_model(self) -> bool:
        return bool(self.paint_priority)

    def name_of(self, value: int) -> str:
        return self.names.get(int(value), f"label_{int(value)}")

    def __len__(self) -> int:
        return len(self.names)
