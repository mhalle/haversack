"""One executor contract, derived from the code that depends on it.

``create_app`` drives whatever executor it is handed - ``LocalExecutor`` in
process, ``ModalExecutor`` on Modal, ``CacheOnlyExecutor`` for the read-only
twin. Nothing made them agree except a hand-written list of six attribute
names in ``test_modal_app.py``, so a seventh requirement added for the local
server reached the deployed side as an ``AttributeError`` in a route, at
runtime, on Modal, where it is most expensive to see. That is the shape of
every bug in this seam: a fix applied to one implementation and not the other.

These tests read the required surface out of ``create_app``'s own AST rather
than restating it, so the list cannot fall behind the code it describes.
"""
import ast
import inspect
import pathlib
import textwrap
import unittest

from haversack import serve


def _executor_surface() -> tuple:
    """Every executor member ``create_app`` touches, split into required and
    getattr-with-a-default.

    Taken from the parameter's real name so renaming it cannot silently empty
    the set - which would turn every assertion below into a tautology.
    """
    tree = ast.parse(pathlib.Path(inspect.getsourcefile(serve)).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "create_app")
    param = fn.args.args[0].arg
    direct = {n.attr for n in ast.walk(fn)
              if isinstance(n, ast.Attribute)
              and isinstance(n.value, ast.Name) and n.value.id == param}
    # create_app also reaches the executor through getattr - a third of the
    # surface. Those all pass a default today, so a missing one degrades rather
    # than 500s, but reading only the direct attributes meant this file's whole
    # premise ("the list cannot fall behind the code") quietly excluded seven
    # names. They are returned separately because they are genuinely optional.
    indirect = {a.args[1].value for a in ast.walk(fn)
                if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
                and a.func.id == "getattr" and len(a.args) >= 2
                and isinstance(a.args[0], ast.Name) and a.args[0].id == param
                and isinstance(a.args[1], ast.Constant)
                and isinstance(a.args[1].value, str)}
    return direct, indirect


def _required_of_executor() -> set:
    """The names create_app uses WITHOUT a fallback - an executor owes all of these."""
    return _executor_surface()[0]


def _optional_of_executor() -> set:
    """Reached through getattr with a default: absent means degraded, not broken."""
    return _executor_surface()[1] - _required_of_executor()


def _declares(cls, name: str) -> bool:
    """Does ``cls`` promise ``name``, as a class attribute or an ``__init__``
    assignment?

    ``hasattr`` alone sees only the first, which would demand a cosmetic class
    attribute for every field a constructor sets - and reading ``__init__``
    alone would miss the plain class attributes. Both forms are real
    declarations; a slot only some *caller* attaches from outside is not, and
    that is the gap worth failing on.
    """
    if hasattr(cls, name):
        return True
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None or not hasattr(init, "__code__"):
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
        except (OSError, SyntaxError):       # pragma: no cover - C or builtin
            continue
        self_name = (tree.body[0].args.args or [None])[0]
        if self_name is None:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == name
                    and isinstance(node.value, ast.Name)
                    and node.value.id == self_name.arg
                    and isinstance(node.ctx, ast.Store)):
                return True
    return False


#: Reached only through the ``if push:`` branch, which an executor opts into by
#: setting ``supports_push = True``. Modal leaves it False and uses the poll
#: branch, so it owes these two nothing - the conditionality is real, not an
#: excuse for a gap.
PUSH_ONLY = {"subscribe", "unsubscribe"}


class ExecutorContract(unittest.TestCase):

    def test_the_optional_surface_is_seen_too(self):
        """A getattr-reached requirement used to be invisible to this file."""
        optional = _optional_of_executor()
        self.assertGreaterEqual(len(optional), 4, f"only found {sorted(optional)}")
        for name in ("supports_push", "content"):
            self.assertIn(name, optional)

    def test_the_derived_surface_is_not_empty(self):
        """The tripwire for the extractor itself: if create_app is refactored so
        the executor is aliased or destructured, this set silently shrinks and
        every other test here starts passing for the wrong reason."""
        req = _required_of_executor()
        self.assertGreaterEqual(len(req), 10, f"only found {sorted(req)}")
        for expected in ("submit", "status_of", "result_file", "new_job_dir"):
            self.assertIn(expected, req)

    def test_both_computing_executors_provide_what_create_app_touches(self):
        from haversack.modal_app import ModalExecutor

        req = _required_of_executor()
        for cls in (serve.LocalExecutor, ModalExecutor):
            need = req - (set() if getattr(cls, "supports_push", False) else PUSH_ONLY)
            missing = sorted(m for m in need if not _declares(cls, m))
            self.assertEqual(missing, [], f"{cls.__name__} is missing {missing}")

    def test_a_push_capable_executor_actually_implements_push(self):
        """supports_push is a claim create_app acts on; make it a checkable one."""
        from haversack.modal_app import ModalExecutor

        for cls in (serve.LocalExecutor, serve.CacheOnlyExecutor, ModalExecutor):
            if getattr(cls, "supports_push", False):
                for m in PUSH_ONLY:
                    self.assertTrue(callable(getattr(cls, m, None)),
                                    f"{cls.__name__} claims push but has no {m}")

    def test_the_shared_methods_take_the_same_arguments(self):
        """Generalises the submit-signature check to every method both provide.

        A parameter added to one side only is a 500 that appears solely on the
        deployed side - how `inputs` first shipped broken. Local is the reference
        because create_app is written against it.
        """
        from haversack.modal_app import ModalExecutor

        checked = 0
        for name in sorted(_required_of_executor() - PUSH_ONLY):
            a = getattr(serve.LocalExecutor, name, None)
            b = getattr(ModalExecutor, name, None)
            if not (callable(a) and callable(b)):
                continue                    # a data attribute, not a method
            pa = set(inspect.signature(a).parameters) - {"self"}
            pb = set(inspect.signature(b).parameters) - {"self"}
            self.assertLessEqual(pa, pb, f"ModalExecutor.{name} is missing "
                                         f"{sorted(pa - pb)}")
            checked += 1
        self.assertGreater(checked, 3, "no shared methods were compared")

    def test_the_read_only_twin_declares_the_gap_it_lives_with(self):
        """CacheOnlyExecutor serves ``create_app(..., read_only=True)`` and has no
        compute path at all, so it legitimately lacks the submit surface. Pinning
        the gap means a NEW requirement lands as a decision rather than as a 500
        on the anonymous twin.
        """
        req = _required_of_executor()
        absent = sorted(m for m in req
                        if not _declares(serve.CacheOnlyExecutor, m))
        self.assertEqual(absent, sorted(EXPECTED_TWIN_GAP),
                         "the read-only twin's surface changed; if that is "
                         "intended, update EXPECTED_TWIN_GAP and say why here")


#: What the twin does without, on purpose: everything behind a compute. Each is
#: unreachable when ``read_only=True``, and by two independent guards - ``authed``
#: returns False outright, and every DELETE route plus the whole /v1/jobs surface
#: is stripped from the router rather than left to 401. If one becomes reachable
#: the twin raises AttributeError instead of returning 404, and the test above
#: fires rather than the anonymous surface 500-ing in production.
#: ``segmenter`` is NOT here: the twin declares the slot (as None) precisely so
#: the weights-version component of a key comes from ``key_fn`` instead - it has
#: no Segmenter to recompute one with, and must key exactly as the writer did.
EXPECTED_TWIN_GAP = {"cache_delete", "cancel", "new_job_dir", "statuses",
                     "submit", "submit_prepare", "subscribe", "unsubscribe"}


if __name__ == "__main__":
    unittest.main()
