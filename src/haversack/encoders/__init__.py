"""Encoders: models whose output is an embedding FIELD - token lattices placed in the patient's
world - rather than a label map (2026-09-23; feldglas docs/embedding-field.md, "Encoding moves into
haversack"). haversack owns the encoders and their weights, beside its segmentation engines, so one
program manages every model's weights; feldglas owns the field format and the receiving end, and is
called here only to write the file.

Names take the task grammar, ``family[.version]:name[@revision]``: ``radar:pretrain``,
``ts.v2:total_fast``. ``segment X`` and ``embed X`` name the same network where both exist - the
verb decides labels or tokens.

The structure is one generic pipeline (:mod:`.pipeline`) and one small module per algorithm family
(:mod:`.radar`, :mod:`.nnunet`), each supplying ``load``, ``prepare`` and ``run`` for the specs in
:mod:`.registry`. A new need is a new spec FIELD, not a branch in the pipeline - the engine rule.

Importing this package pulls in no torch: specs are data, and a family module is imported only when
an embedding runs.
"""
from .registry import ALIASES, ENCODERS, EncoderSpec, LatticeSpec, WeightsFile, resolve

__all__ = ["ALIASES", "ENCODERS", "EncoderSpec", "LatticeSpec", "WeightsFile", "resolve"]
