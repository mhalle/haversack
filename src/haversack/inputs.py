"""Inputs by reference: the public door to haversack's standard form of input (2026-09-26).

The cached form IS the standard form (the user's decision): every input - a hosted source, an
upload's digest, a local file or folder - is stored once, as its input copy where the reader
can read it (docs/input-copy.md), and read from there by haversack's engines and by anyone
else's program alike. This module is how a program outside haversack reads it, without
parsing DICOM again and without knowing where the cache keeps anything:

    from haversack import inputs
    x = inputs.open("idc:4682f41a-65d7-4a7b-8050-952f73abb746")   # or a path, or sha256:...
    x.identity          # what names it: the source identifier, or the digest of its bytes
    x.image()           # a SimpleITK image, geometry resolved
    x.array((100, 132)) # slices 100..131 only - a compressed copy decodes just their chunks
    x.tags()            # its DICOM tags as JSON: {"series": {...}, "slices": [{...}, ...]}
    x.record            # where it came from: origin, license, citation, the bytes' digest

A path under the cache is not the interface: eviction moves things, and ``open`` is what
holds an input for the caller. The store form is used when ``HAVERSACK_INPUT_STORE=blobs``
(until that is the default); without it, ``open`` reads what the legacy cache holds.
"""
from __future__ import annotations

from pathlib import Path


class Input:
    """One input, stored and held. Cheap to make: nothing is read until it is asked for."""

    def __init__(self, identity: str | None, path: Path, record: dict | None):
        self.identity = identity
        self.path = Path(path)
        self.record = record

    def __repr__(self) -> str:
        return f"Input({self.identity or str(self.path)!r})"

    @property
    def is_copy(self) -> bool:
        from .input_copy import is_copy
        return is_copy(self.path) or self.path.name.endswith(".duckn.zip")

    def image(self):
        """The whole input as a SimpleITK image - read exactly as haversack's engines read it."""
        from . import io
        return io.read_image(self.path)

    def array(self, slices: tuple[int, int] | None = None):
        """The voxels as a numpy array in (Z, Y, X) order, all of them or ``slices`` = (start,
        stop) along the slice axis. From a copy only what is asked for is read: the chunks
        holding those slices (compressed), or the slices themselves (uncompressed, mapped)."""
        import numpy as np
        if self.is_copy:
            from .input_copy import _layout
            meta, how, start, dt, shape = _layout(self.path)
            lo, hi = (0, shape[0]) if slices is None else slices
            if how == "zstd":
                import zarr
                from zarr.storage import ZipStore
                store = ZipStore(str(self.path), mode="r")
                try:
                    return np.asarray(zarr.open_array(store, mode="r")[lo:hi])
                finally:
                    store.close()
            import mmap
            import builtins
            with builtins.open(self.path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    plane = int(np.prod(shape[1:]))
                    raw = np.frombuffer(mm, dtype=dt, count=(hi - lo) * plane,
                                        offset=start + lo * plane * np.dtype(dt).itemsize)
                    out = raw.reshape((hi - lo,) + tuple(shape[1:])).copy()
                    del raw                    # no view may outlive the map it points into
                finally:
                    mm.close()
            return out
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(self.image())
        return arr if slices is None else arr[slices[0]:slices[1]]

    def tags(self) -> dict:
        """The DICOM tags the copy carries, as JSON: ``{"series": {keyword: value},
        "slices": [{keyword: value}, ...]}`` in duckn's dicom encoding - empty for an input
        that is not DICOM, or not a copy. (Sourced from SimpleITK's dictionary today; the
        header design in docs/cache-consolidation.md moves it to pydicom, fuller, same shape.)"""
        if not self.is_copy:
            return {}
        from .input_copy import _duckn, _layout
        attrs = _duckn(_layout(self.path)[0])
        series = ((attrs.get("extensions") or {}).get("dicom") or {}).get("tags") or {}
        axes = attrs.get("axes") or []
        samples = (axes[0].get("samples") or []) if axes else []
        slices = [((s.get("metadata") or {}).get("dicom") or {}) for s in samples]
        return {"series": series, "slices": slices} if (series or any(slices)) else {}


def open(spec, *, cache_dir=None) -> Input:            # noqa: A001 - the module's verb
    """Store ``spec`` if it is not stored yet, and hold it for the caller: a hosted source
    (``idc:...``, anything ``haversack sources`` lists), a local file or folder, or the
    digest of an upload this machine holds."""
    from . import sources
    from .content import is_digest
    from .inputstore import command_inputs, input_store_enabled
    spec = str(spec)
    parsed = sources.parse_input(spec)
    if is_digest(spec) and input_store_enabled():
        store = command_inputs(Path(cache_dir) / "store" if cache_dir else sources.default_input_store())
        if not store.has(spec):
            from .inputstore import InputGone
            raise InputGone(f"{spec} is not held by this machine's input store")
        return Input(spec, store.get_or_fetch(spec, fetch=None), store.record(spec))
    if parsed is None and input_store_enabled() and Path(spec).exists():
        store = command_inputs(Path(cache_dir) / "store" if cache_dir else sources.default_input_store())
        digest, path = store.ingest_with_identity(Path(spec))
        return Input(digest, path, sources.input_record(spec))
    path = sources.materialize(spec, cache_dir=cache_dir)
    identity = f"{parsed[0]}:{parsed[1]}" if parsed else None
    return Input(identity, path, sources.input_record(spec, cache_dir=cache_dir))
