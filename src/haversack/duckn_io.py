"""duckn volumes as haversack input.

INTERNAL AND UNSTABLE, undocumented on purpose (see ``ranked_store``): the duckn format and
this package are iterated together.

A duckn volume is a zarr v3 array - in a directory or a standard zarr zip - whose attributes
carry the geometry (``space``, ``space_origin``, per-axis ``space_direction`` and centering)
and the value transforms (a CT stored as uint16 with a slope and intercept). Reading one for
the pipeline means turning that into the SimpleITK image every other input becomes. The
conversion is duckn's own: its reader parses the metadata through its models and applies the
value transforms, and its SimpleITK adapter does the axis reversal, the RAS-to-LPS flip and
the direction cosines. Re-deriving any of that here would be a second, worse copy of the
standard, and axis conventions are exactly where silent geometry bugs live.

duckn is imported lazily and is not a core dependency; the dev env installs it from the
sibling checkout.
"""
from __future__ import annotations

from pathlib import Path

from .errors import InputError

STORE_SUFFIXES = (".duckn.zip", ".zarr.zip", ".duckn", ".zarr")


def is_duckn_store(path) -> bool:
    """Whether ``path`` names a duckn/zarr volume: by suffix, or a directory holding a
    ``zarr.json`` (which a DICOM series directory never does)."""
    p = Path(path)
    name = p.name.lower()
    if any(name.endswith(s) for s in STORE_SUFFIXES):
        return True
    return p.is_dir() and (p / "zarr.json").exists()


def read_duckn_image(path):
    """Read a duckn volume as a 3-D SimpleITK image in LPS, values calibrated (value
    transforms applied), geometry from the duckn attributes."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"input not found: {p}")
    try:
        from duckn import read_duckn
        from duckn.sitk_adapter import to_sitk
        from duckn.volume import Volume
        from zarr.storage import LocalStore, ZipStore
    except ImportError as e:                       # pragma: no cover - env without duckn
        raise InputError(f"{p} is a duckn/zarr volume, and duckn is not installed") from e
    # The container is picked here, by our suffixes: duckn's own path-based opener keys on
    # `.zarr.zip` alone, so a `.duckn.zip` handed to it as a path would open as a directory.
    # Its reader takes a store object, which is the seam that avoids that.
    store = (ZipStore(str(p), mode="r") if p.name.lower().endswith(".zip")
             else LocalStore(str(p), read_only=True))
    try:
        data, meta = read_duckn(store)                 # duckn's models parse the attributes
        vol = Volume(raw=data, metadata=meta)          # .data applies the value transforms
    except Exception as e:                          # noqa: BLE001 - duckn's message is the finding
        raise InputError(f"cannot read {p} as a duckn volume: {e}") from e
    finally:
        if hasattr(store, "close"):
            store.close()
    spatial = [ax for ax in (vol.metadata.axes or []) if ax.kind is None or ax.kind == "space"]
    if vol.raw.ndim != 3 or len(spatial) != vol.raw.ndim:
        raise InputError(f"expected a 3-D scalar volume; {p} has shape {tuple(vol.raw.shape)} "
                         f"with {len(spatial)} spatial axes")
    return to_sitk(vol)
