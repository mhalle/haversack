"""Ranked stores on disk: the container, and the metadata that goes in it.

INTERNAL AND UNSTABLE. Deliberately undocumented - not in the README, the CHANGELOG or the
CLI - so the layout can move while duckn and this package are iterated together with no
outside reader to keep compatible. The tools under ``tools/ranked_*.py`` are its callers.

Two responsibilities, kept small on purpose:

**The container.** A ranked store is a zarr v3 hierarchy. It lives either in a directory or
in a single **standard zarr zip file**: the same hierarchy as archive entries at the archive
root, stored uncompressed because every chunk is already zstd - exactly what zarr's own
``ZipStore`` writes and reads, so any zarr reader opens it without knowing anything about
this package. :func:`open_store` picks the container by the path's suffix and returns a
handle; arbitrary keys such as the format ``README.md`` go through the store too, never
through the filesystem, so the two containers hold identical keys.

A zip archive cannot rewrite an entry - writing an existing key appends a duplicate - and
zarr rewrites a group's ``zarr.json`` every time its attributes change, which a build does
several times per group. So a zip is never written through ``ZipStore``: writing and amending
go through a **staging directory** packed into the archive on :meth:`RankedStore.close`
(atomically, beside the target), and reading goes through ``ZipStore``. The archive is then
byte-for-byte the directory store, zipped, with no duplicate entries, and every in-place tool
works on a zip exactly as on a directory.

**The metadata.** The attributes are duckn: geometry (``space``, ``space_origin``, ``axes``
with a centering per axis), the ``seg`` extension for the segments, ``provenance`` for what
produced the store, and this package's own ``ranked`` and ``haversack`` extension blocks. They
are built through duckn's own models (``DucknMetadata``, ``AxisMetadata``, ``Segment``,
``SegmentationExtension``) and serialized by duckn's own ``duckn_attrs``, and read back through
the same models with duckn's validators. Hand-assembling those dicts is how three metadata
bugs were written into shipped stores; the models are the standard, and every field they
refuse is a bug caught at write time instead of at a reader. The extension blocks that are
ours have no model and stay dicts - inside ``extensions``, where duckn says private blocks go.

zarr and duckn are imported lazily: the package must import without either (they are not
core dependencies; the dev env installs duckn from the sibling checkout).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .errors import InputError

DUCKN_VERSION = "1.0"
SPACE = "left-posterior-superior"
ZIP_SUFFIX = ".zip"

__all__ = ["RankedStore", "open_store", "is_zip", "grid_metadata", "grid_attrs",
           "grid_reference", "brick_attrs", "part_attrs", "leaf", "group", "segmentation",
           "root_attrs", "read_metadata", "validate_array", "read_segmentation"]


def is_zip(path) -> bool:
    """Whether ``path`` names a zip container (by suffix; the file need not exist yet)."""
    return str(path).lower().endswith(ZIP_SUFFIX)


class RankedStore:
    """An open store: ``root`` is the zarr group, ``store`` the zarr store behind it.

    Use as a context manager, or call :meth:`close`. A zip (any mode) and a directory
    being created are built in a staging directory; ``close`` packs or moves it onto the
    target, and leaving the ``with`` block on an exception discards the staging and leaves
    the target untouched. A directory opened with ``"a"`` is the exception: it is amended
    in place, and a failure part-way leaves what was written.
    """

    def __init__(self, path, store, root, mode: str, staging: Path | None = None, lock=None):
        self.path = Path(path)
        self.store = store
        self.root = root
        self.mode = mode
        self.is_zip = is_zip(path)
        self._staging = staging
        self._lock = lock
        self._closed = False

    # -- arbitrary keys (the README travels as a key, never as a file beside the store) --

    def exists(self, key: str) -> bool:
        from zarr.core.sync import sync
        return bool(sync(self.store.exists(key)))

    def read_text(self, key: str) -> str:
        from zarr.core.buffer import default_buffer_prototype
        from zarr.core.sync import sync
        buf = sync(self.store.get(key, default_buffer_prototype()))
        if buf is None:
            raise KeyError(key)
        return buf.to_bytes().decode("utf-8")

    def write_text(self, key: str, text: str) -> None:
        from zarr.core.buffer import default_buffer_prototype
        from zarr.core.sync import sync
        if self.mode == "r":
            raise ValueError(f"{self.path}: opened read-only")
        proto = default_buffer_prototype()
        sync(self.store.set(key, proto.buffer.from_bytes(text.encode("utf-8"))))

    def size_bytes(self) -> int:
        """Bytes on disk: the archive's size once packed, else the directory's files."""
        if self.is_zip and self._staging is None:
            return self.path.stat().st_size
        root = self._staging if self._staging is not None else self.path
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())

    def close(self, *, discard: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                if hasattr(self.store, "close"):
                    self.store.close()
                if self._staging is not None and not discard:
                    # the target changes only here, after a complete build: a failure
                    # anywhere above leaves whatever was at the path exactly as it was
                    if self.is_zip:
                        _pack(self._staging, self.path)
                    else:
                        _replace_dir(self._staging, self.path)
            finally:
                if self._staging is not None:
                    shutil.rmtree(self._staging, ignore_errors=True)
                    self._staging = None
        finally:
            if self._lock is not None:
                self._lock.release()
                self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        self.close(discard=exc_type is not None)

    def __repr__(self):
        kind = "zip" if self.is_zip else "dir"
        return f"RankedStore({self.path.name}, {kind}, mode={self.mode!r})"


def _pack(src_dir: Path, archive: Path) -> None:
    """Zip a directory store: every file as an entry at its relative posix path, stored
    uncompressed, in sorted order. Written beside the target and moved into place."""
    partial = archive.with_name(archive.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in sorted(p for p in src_dir.rglob("*") if p.is_file()):
            zf.write(f, f.relative_to(src_dir).as_posix())
    os.replace(partial, archive)


def _replace_dir(src: Path, dst: Path) -> None:
    """Move a finished staging directory onto ``dst``; a previous ``dst`` goes away only
    once the new one is in place."""
    old = None
    if dst.exists():
        old = dst.with_name(f"{dst.name}.old-{os.getpid()}")
        os.replace(dst, old)
    try:
        os.replace(src, dst)
    except OSError:
        if old is not None:
            os.replace(old, dst)
        raise
    if old is not None:
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink(missing_ok=True)


class _Lock:
    """One writer per store: an advisory ``flock`` on a ``<store>.lock`` file beside the
    target, held for the life of the handle. A second writer is refused at open, not
    discovered as an interleaved archive at close."""

    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    @classmethod
    def acquire(cls, target: Path) -> "_Lock":
        lock = cls(target.with_name(target.name + ".lock"))
        lock.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl
        except ImportError:                       # no advisory locks on this platform
            lock.fd = os.open(lock.path, os.O_CREAT | os.O_RDWR, 0o644)
            return lock
        for _ in range(20):
            fd = os.open(lock.path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise InputError(f"{target}: another process is writing this store "
                                 f"(lock {lock.path.name}); wait for it, or remove the lock "
                                 "file if that process is gone") from None
            # The releasing writer unlinks its lock file. A lock taken on an inode that
            # was unlinked between our open and our flock guards nothing - a third
            # writer creates a fresh file at the path and locks that - so the lock
            # counts only when the path still names the inode we hold.
            try:
                same = os.stat(lock.path).st_ino == os.fstat(fd).st_ino
            except FileNotFoundError:
                same = False
            if same:
                lock.fd = fd
                return lock
            os.close(fd)
        raise InputError(f"{target}: could not take the store lock ({lock.path.name})")

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        os.close(self.fd)
        self.fd = None


def _looks_like_store(p: Path) -> bool:
    """Whether what sits at ``p`` is a zarr group (a ranked store, or at least something
    this module wrote); anything else is not ours to replace."""
    if is_zip(p):
        try:
            with zipfile.ZipFile(p) as zf:
                return "zarr.json" in zf.namelist()
        except (zipfile.BadZipFile, OSError):
            return False
    return p.is_dir() and (p / "zarr.json").is_file()


def open_store(path, mode: str = "r") -> RankedStore:
    """Open a ranked store at ``path``: a directory, or a zarr zip when it ends in ``.zip``.

    ``mode``: ``"r"`` read; ``"w"`` create, replacing a store already there (never a
    directory, archive or file that is not one); ``"a"`` open an existing store to amend. A zip, and
    a directory being created, are worked on in a staging directory beside the target and
    moved onto it when the handle closes without error; leaving the ``with`` block on an
    exception discards the staging and the target is exactly what it was. A directory
    opened with ``"a"`` is amended in place. Writers hold a lock on the target.
    """
    try:
        import zarr
        from zarr.storage import LocalStore, ZipStore
    except ImportError as e:
        raise InputError("ranked stores need zarr (the duckn extra): "
                         "uv sync --extra duckn, or uv pip install zarr") from e

    p = Path(path)
    if mode != "r" and p.is_symlink():
        # write through the link: the staging must sit beside the REAL store, and the
        # replace must land there rather than turn the link into a directory
        p = Path(os.path.realpath(p))
    zipped = is_zip(p)
    if mode == "r":
        if not p.exists():
            raise FileNotFoundError(str(p))
        st = ZipStore(str(p), mode="r") if zipped else LocalStore(str(p), read_only=True)
        return RankedStore(p, st, zarr.open_group(store=st, mode="r"), mode)
    if mode not in ("w", "a"):
        raise ValueError(f"mode must be 'r', 'w' or 'a', not {mode!r}")
    if mode == "a" and not p.exists():
        raise FileNotFoundError(str(p))
    if p.exists() and not _looks_like_store(p):
        # a directory of photos, a stray archive, a text file: whatever it is, it is
        # not ours to replace
        raise InputError(f"{p}: exists and is not a ranked store; refusing to "
                         f"{'replace' if mode == 'w' else 'amend'} it")
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = _Lock.acquire(p)
    staging = None
    try:
        if zipped or mode == "w":
            staging = Path(tempfile.mkdtemp(prefix=p.name + ".staging-", dir=p.parent))
            if zipped and mode == "a":
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(staging)
            where = staging
        else:
            where = p
        st = LocalStore(str(where))
        root = (zarr.create_group(store=st) if mode == "w"
                else zarr.open_group(store=st, mode="r+"))
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock.release()
        raise
    return RankedStore(p, st, root, mode, staging=staging, lock=lock)


# ----------------------------------------------------------------------------------------
# Writing metadata: duckn's models, duckn's serializer.
# ----------------------------------------------------------------------------------------

def _axes(direction_xyz, spacing_zyx, *, list_axis: bool, centering: str):
    from duckn import AxisMetadata
    if len(spacing_zyx) != 3:
        raise ValueError(f"three spacings for a 3-D grid, got {list(spacing_zyx)}")
    D = np.asarray(direction_xyz, float).reshape(3, 3)
    cols = [D[:, 2], D[:, 1], D[:, 0]]                              # array axes Z, Y, X
    axes = [AxisMetadata(kind="space", centering=centering, unit="mm",
                         space_direction=[round(float(v), 9) for v in (c * s)])
            for c, s in zip(cols, spacing_zyx)]
    return ([AxisMetadata(kind="list")] + axes) if list_axis else axes


def grid_metadata(direction_xyz, spacing_zyx, origin_xyz, *, list_axis: bool = False,
                  centering: str = "cell"):
    """The duckn geometry of an array on a 3-D grid, as a ``DucknMetadata``.

    ``direction_xyz`` are the row-major direction cosines, ``spacing_zyx`` the spacing per
    ARRAY axis, ``origin_xyz`` the world position of voxel 0's sample. ``list_axis`` prepends
    the ``list`` axis of a rank/support/pair array. ``centering`` is duckn's word for what a
    sample represents - ``node`` for a corner-rule grid, ``cell`` for an image grid or a
    center-rule grid - and it is what a resampler holds fixed, so it is never a default here.
    """
    from duckn import DucknMetadata
    return DucknMetadata(version=DUCKN_VERSION, space=SPACE,
                         space_origin=[round(float(v), 6) for v in origin_xyz],
                         axes=_axes(direction_xyz, spacing_zyx, list_axis=list_axis,
                                    centering=centering))


def grid_attrs(direction_xyz, spacing_zyx, origin_xyz, *, list_axis: bool = False,
               centering: str = "cell") -> dict:
    """:func:`grid_metadata` serialized as an array's ``attributes`` (re-validated)."""
    from duckn.models import duckn_attrs
    return duckn_attrs(grid_metadata(direction_xyz, spacing_zyx, origin_xyz,
                                     list_axis=list_axis, centering=centering))


def grid_reference(direction_xyz, spacing_zyx, origin_xyz, shape_zyx, *,
                   centering: str = "cell") -> dict:
    """A grid REFERRED TO from an extension block (a restore target, a source grid).

    The same vocabulary as an array's own geometry plus ``samples``, the shape the grid would
    have - which an array states through its shape and a reference must spell out.
    """
    m = grid_metadata(direction_xyz, spacing_zyx, origin_xyz, centering=centering)
    d = m.model_dump(exclude_none=True, exclude={"version"})
    d["samples"] = [int(v) for v in shape_zyx]
    return d


def brick_attrs(direction_xyz, spacing_zyx, origin_xyz, brick: int, *, list_axis: bool = True
                ) -> dict:
    """Geometry of a brick summary (the occupancy index): a grid ``brick`` times coarser
    whose samples are cell centres, with a ``list`` axis for the class.

    The last brick along an axis is partial when the shape is not a multiple of ``brick``,
    so its true centre is nearer than this uniform grid says; left as-is deliberately - the
    array is a conservative index, not a measurement, and a uniform grid keeps it a readable
    duckn array rather than a private layout.
    """
    D = np.asarray(direction_xyz, float).reshape(3, 3)
    off = np.asarray([(brick - 1) / 2 * spacing_zyx[2], (brick - 1) / 2 * spacing_zyx[1],
                      (brick - 1) / 2 * spacing_zyx[0]], float)
    origin = np.asarray(origin_xyz, float) + D @ off
    return grid_attrs(direction_xyz, [s * brick for s in spacing_zyx], origin,
                      list_axis=list_axis, centering="cell")


def part_attrs(ranked_block: dict) -> dict:
    """A part group's attributes: the ``ranked`` extension block, nothing else."""
    from duckn import DucknMetadata
    from duckn.models import duckn_attrs
    return duckn_attrs(DucknMetadata(version=DUCKN_VERSION,
                                     extensions={"ranked": ranked_block}))


def leaf(id: str, name: str, label_value: int, *, layer: int | None = None,
         extent: list[int] | None = None, background: bool = False,
         color: list[float] | None = None):
    """A duckn leaf ``Segment``: one label value in one layer. ``background`` marks the
    layer's background leaf (seg spec 0.7 §3.2)."""
    from duckn import Segment
    return Segment(id=id, name=name, label_value=int(label_value), layer=layer,
                   extent=extent, background=(True if background else None), color=color)


def group(id: str, name: str, members: list[str], *, disjoint: bool = False,
          exhaustive: bool = False, color: list[float] | None = None):
    """A duckn group ``Segment``: the union of ``members`` (segment ids). ``disjoint``
    claims the members share no voxel; ``exhaustive`` claims they exhaust the thing the
    group names; both make a partition (seg spec 0.7 §2)."""
    from duckn import Segment
    return Segment(id=id, name=name, members=list(members),
                   disjoint=(True if disjoint else None),
                   exhaustive=(True if exhaustive else None), color=color)


_SCT = {"name": "SNOMED CT", "url": "http://snomed.info/sct",
        "url_template": "http://snomed.info/id/{code}"}


def segmentation(segments, *, terminologies: dict | None = None):
    """The ``seg`` extension over ``segments``, validated by duckn's consistency rules."""
    from duckn import SEG_EXTENSION_VERSION, SegmentationExtension, TerminologyEntry
    from duckn import validate_seg_extension
    terms = {k: TerminologyEntry(**v) for k, v in (terminologies or {"SCT": _SCT}).items()}
    ext = SegmentationExtension(version=SEG_EXTENSION_VERSION, terminologies=terms,
                                segments=list(segments))
    validate_seg_extension(ext)
    return ext


def root_attrs(seg_ext, **extensions) -> dict:
    """The root group's attributes: the validated ``seg`` extension plus this package's own
    blocks (``haversack``, ``provenance``, ...), serialized by duckn."""
    from duckn import DucknMetadata
    from duckn.models import duckn_attrs
    ext = {"seg": seg_ext.model_dump(exclude_none=True)}
    ext.update(extensions)
    return duckn_attrs(DucknMetadata(version=DUCKN_VERSION, extensions=ext))


# ----------------------------------------------------------------------------------------
# Reading metadata: the same models, duckn's validators.
# ----------------------------------------------------------------------------------------

def read_metadata(node):
    """The ``duckn`` attributes of a zarr group or array, parsed by duckn's model - so a key
    duckn does not know, or a geometry it rejects, raises here rather than misleading a
    reader downstream."""
    from duckn import DucknMetadata
    return DucknMetadata.model_validate(node.attrs.asdict().get("duckn", {}))


def validate_array(arr) -> None:
    """duckn's own shape-consistency check of an array's geometry against its shape."""
    from duckn import validate_against_shape
    validate_against_shape(read_metadata(arr), tuple(int(s) for s in arr.shape))


def read_segmentation(root):
    """The root's ``seg`` extension as a validated ``SegmentationExtension``."""
    from duckn import SegmentationExtension, validate_seg_extension
    ext = (read_metadata(root).extensions or {}).get("seg")
    if ext is None:
        raise KeyError("no seg extension on the root group")
    seg = SegmentationExtension.model_validate(ext)
    validate_seg_extension(seg)
    return seg
