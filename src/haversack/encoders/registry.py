"""The encoders haversack can run, as data: what each is, where its weights come from, how its
input is prepared, what its lattices are. No torch here - :mod:`.pipeline` imports a family module
only to encode.

A spec states facts, measured or pinned, and nothing else: an unmeasured reach or look offset is
absent (``None``), never guessed, because it reaches the field's ``thickness`` and ``support`` and
a client acts on those.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import InputError


@dataclass(frozen=True)
class WeightsFile:
    """One weights file, pinned: where it is published, at which revision, and its digest."""

    url: str                        # the pinned download URL (a revision, never a branch)
    sha256: str
    size: int
    name: str                       # the file's name under the encoder's weights directory
    source: str = ""                # human-readable origin, e.g. "huggingface: radar-generalist/RADAR"


@dataclass(frozen=True)
class LatticeSpec:
    """One lattice of an encoder's field."""

    layer: str                                      # the encoder layer it is (part of the comparability key)
    kernel: tuple[int, int, int]                    # model voxels per token, (Z, Y, X)
    width: int                                      # channels
    receptive_mm: tuple[float, float, float] | None = None       # measured reach, full width; None: not measured
    support_offset_mm: tuple[float, float, float] | None = None  # evidence minus drawn center; None: not measured


@dataclass(frozen=True)
class EncoderSpec:
    name: str                                       # family[.version]:name
    family: str                                     # the module under haversack.encoders that runs it
    description: str
    revision: str                                   # what pins the weights (a commit, a release tag)
    lattices: tuple[LatticeSpec, ...]
    attribution: str                                # the key in data/attribution.json (ecosystems)
    license: str                                    # the weights' license - the field inherits it
    weights: tuple[WeightsFile, ...] = ()           # files this encoder downloads; () when another catalog's
    uses_task: str | None = None                    # an nnU-Net encoder: the segmentation task whose network it is
    modality: str = "CT"                            # what it was trained on (which of its makers' papers apply)
    stage: str = "raw"
    metric: str = "cosine"
    normalized: bool = False
    options: dict = field(default_factory=dict)     # family-specific facts (architecture, input convention)

    @property
    def family_name(self) -> str:
        return self.name.split(":", 1)[0]


# RADAR (Alibaba DAMO; github.com/alibaba-damo-academy/RADAR, Apache-2.0 code; weights on Hugging
# Face under CC BY-NC-SA 4.0). Its vision encoder is a standard nnU-Net PlainConvEncoder plus three
# 1x1 projections - built here from the installed dynamic_network_architectures, bit-identical to
# upstream's own on the real checkpoint (2026-09-23), so no upstream code is carried. The facts
# below are the RADAR study's (medseg docs/radar-idc-validation/EXPLORATION.md, feldglas):
# lattices deep / mid / fine; reach ~80 / 40 / 20 mm (the measured point spread, section 2); look
# offsets from the phantom (feldglas adapters/radar.py LOOK_OFFSET_MM), negative: toward index 0.
_RADAR_REPO = "radar-generalist/RADAR"
_RADAR_REVISION = "3861f2d3e004451c77a6d3afc3e66a9c3e00bc60"
_RADAR_ARCH = {
    "n_stages": 6, "features_per_stage": [32, 64, 128, 256, 320, 320],
    "kernel_sizes": [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "strides": [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
}
_RADAR_LATTICES = (
    LatticeSpec("deep", (8, 32, 32), 256, (80.0, 80.0, 80.0), (-6.6, -6.3, -7.6)),
    LatticeSpec("mid", (4, 16, 16), 256, (40.0, 40.0, 40.0), (-2.8, -2.2, -2.7)),
    LatticeSpec("fine", (2, 8, 8), 256, (20.0, 20.0, 20.0), (-1.1, -1.6, -1.4)),
)


def _radar(name: str, filename: str, sha256: str, size: int, description: str) -> EncoderSpec:
    return EncoderSpec(
        name=name, family="radar", description=description, revision=_RADAR_REVISION,
        lattices=_RADAR_LATTICES, attribution="radar", license="CC-BY-NC-SA-4.0",
        weights=(WeightsFile(url=f"https://huggingface.co/{_RADAR_REPO}/resolve/{_RADAR_REVISION}/{filename}",
                             sha256=sha256, size=size, name=filename, source=f"huggingface: {_RADAR_REPO}"),),
        options={"arch": _RADAR_ARCH, "prefix": "visual_encoder.",
                 # projections, deep to fine: (checkpoint key, input channels, which skip from the last)
                 "projections": (("proj1", 320, 1), ("proj2", 320, 2), ("proj3", 256, 3)),
                 # the input convention, as upstream's DataFolder: LAS, trilinear to 1 x 1 x 5 mm with no
                 # prefilter, clip to [-300, 400] HU, min-max to [0, 1], crop the non-air box with a
                 # (5, 20, 20)-voxel margin, pad at the end to multiples of 32 and at least (96, 256, 384)
                 "orientation": "LAS", "spacing_mm": (1.0, 1.0, 5.0), "window_hu": (-300.0, 400.0),
                 "crop_margin": (5, 20, 20), "pad_multiple": 32, "min_shape": (96, 256, 384),
                 "max_resampled": 1000})


def _nnunet(name: str, dataset: int, stages, kernels, widths, align: int, description: str) -> EncoderSpec:
    """An nnU-Net network's encoder: its skips at ``stages`` as lattices. The weights are its
    task's (``haversack weights fetch <task>``); reach and look offset are not measured, so absent.
    The license is TotalSegmentator's (Apache-2.0 for these tasks)."""
    return EncoderSpec(
        name=name, family="nnunet", description=description, revision=f"Dataset{dataset}",
        lattices=tuple(LatticeSpec(f"encoder.stages.{s}", k, w) for s, k, w in zip(stages, kernels, widths)),
        attribution="ts.v2", license="Apache-2.0", uses_task=name,
        options={"dataset": dataset, "stages": tuple(stages), "align": align})


ENCODERS: dict[str, EncoderSpec] = {s.name: s for s in (
    _radar("radar:pretrain", "checkpoint_radar_pretrain.pth",
           "5e8b1b50b92162fcafcf8231e51ffbe9d1be2be52ccff4f967e4beb71e4499f4", 1566049482,
           "RADAR's pretrained vision encoder (checkpoint_radar_pretrain): three lattices of 256 channels, "
           "10 / 20 / 40 mm tokens - the checkpoint of the RADAR study"),
    # the RADAR study's "null model" (EXPLORATION 5.12): an encoder trained only to label anatomy
    _nnunet("ts.v2:total_fast", 297, (2, 3, 4), ((4, 4, 4), (8, 8, 8), (16, 16, 16)), (128, 256, 320), 16,
            "TotalSegmentator total_fast's encoder (Dataset 297, 3 mm): stages 2-4, 12 / 24 / 48 mm tokens"),
    _nnunet("ts.v2:total", 291, (3, 4, 5), ((8, 8, 8), (16, 16, 16), (32, 32, 32)), (256, 320, 320), 32,
            "TotalSegmentator total's organs model's encoder (Dataset 291, 1.5 mm; total's first of five "
            "parts): stages 3-5, 12 / 24 / 48 mm tokens"),
)}

# names fields were written under before the task grammar (feldglas <= 0.1.0): a field's
# comparability key carries them, so they keep resolving
ALIASES: dict[str, str] = {"radar": "radar:pretrain", "null-totalsegmentator": "ts.v2:total_fast",
                           "null-totalsegmentator-1.5mm": "ts.v2:total"}


def resolve(name: str) -> EncoderSpec:
    """The spec for ``name`` (``family:name``, an alias, optionally ``@revision``). A pinned
    revision must be the one this haversack runs - an encoder has one revision at a time."""
    base, _, pin = str(name).partition("@")
    base = ALIASES.get(base, base)
    spec = ENCODERS.get(base)
    if spec is None:
        near = sorted(n for n in ENCODERS if n.split(":")[0] == base.split(":")[0]) or sorted(ENCODERS)
        raise InputError(f"no encoder {name!r}; haversack encodes with: {', '.join(near)}")
    if pin and not spec.revision.startswith(pin):
        raise InputError(f"{name!r}: this haversack runs {spec.name} at revision {spec.revision[:12]}, not {pin}")
    return spec
