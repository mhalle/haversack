"""The errors haversack raises deliberately.

A consumer needs to tell "these weights are not downloaded yet" (retry after fetching) from
"haversack cannot run this model" (never retry, report it) from "this image is unusable" (the user's
fault) - today that means matching on message text. Each class therefore also inherits the
builtin it replaces, so ``except FileNotFoundError`` keeps working and ``except ModelNotFound``
becomes possible; callers can catch at whichever level suits them.
"""
from __future__ import annotations


class HaversackError(Exception):
    """Base for every error haversack raises on purpose. Catch this to catch all of them."""


class InputError(HaversackError, ValueError):
    """The image or the arguments are unusable: not 3D, unreadable, contradictory options."""


class RequestError(InputError):
    """A request that can be refused before any work starts: an unknown
    parameter, a missing input role, a role bound twice.

    Carries a machine-readable ``code`` plus the fields a client needs in order
    to fix the request. The point is that the API answers in ONE error
    vocabulary: without this, the shape of a rejection would depend on which
    validator happened to catch it, and a client would have to learn pydantic's
    ``loc``/``msg``/``type`` alongside ours.
    """

    #: The HTTP status the wire answers with. 422: the request itself is wrong.
    status = 422

    def __init__(self, code: str, message: str, **detail):
        super().__init__(message)
        self.code = code
        self.detail = {"code": code, "message": message, **detail}


class UnresolvedReference(RequestError):
    """A well-formed request that refers to something this server does not hold NOW: a
    result that was never computed here, has been evicted, or was republished with other
    bytes since the caller pinned it (``result:`` references, 2026-09-20).

    409, not 422: nothing about the request's shape is wrong, and the same request
    succeeds once the state is put right - which is RFC 9110's conflict, and what a pinned
    task version this server does not run already answers. Raised at submit, where the
    caller is still listening, and again by a worker that re-resolves the reference and
    finds other bytes: the job fails by this name rather than compute from them.
    """

    status = 409


class ModelNotFound(HaversackError, FileNotFoundError):
    """A model, configuration, fold or weights file is not where it should be.

    Recoverable: fetch the weights (``haversack.weights_fetch``) or point ``model_root`` elsewhere
    and try again.
    """


class AmbiguousModel(ModelNotFound):
    """A dataset holds more than one model folder that fits what was asked, and nothing said
    which one to run.

    A ``Dataset<id>_*`` directory is ``<trainer>__<plans>__<configuration>/`` folders, and
    TotalSegmentator's v3 release (2026-09) ships TWO ``3d_fullres`` folders per dataset in
    831-836: ``nnUNetPlans`` (what upstream runs) and ``nnUNetResEncUNetLPlans_8`` (its
    ``model_size="small"`` only). Keying the folders by configuration alone made the second
    replace the first, so the small model ran under the default's name with nothing said.
    The remedy is to state ``plans=`` (and ``trainer=`` if that too is shared) - never to
    choose. A subclass of :class:`ModelNotFound` so every door that already reports an
    installed-but-unresolvable model (``info()``'s ``unresolved``) reports this the same way.
    """


class UnsupportedModel(HaversackError, NotImplementedError):
    """A valid nnU-Net model that haversack cannot run *yet*.

    Not the caller's mistake and not recoverable by retrying - e.g. a non-identity
    ``transpose_forward``, region-based (sigmoid) labels, multi-channel input, or a
    ``3d_cascade_fullres`` configuration. Distinct from :class:`ModelNotFound` so a UI can say
    "this model is not supported" rather than "file missing".
    """


class ResourceError(HaversackError, RuntimeError):
    """Out of device or host memory, after haversack's own fallbacks were exhausted."""


class Cancelled(HaversackError):
    """The caller's cancel signal fired; the run stopped between patches."""
