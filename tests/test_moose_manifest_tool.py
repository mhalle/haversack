"""tools/gen_moose_manifest.py: the one generator that never reads the assets it offers.

The others read each zip's layout by Range, so a dead URL stops them by itself.
This one parses a Python registry, and so it must check what it records: MOOSE's
``clin_ct_dental`` asset was reported dead (403) when the installer, sending
Python's default User-Agent, was what the host refused. The check sends the
installer's kind of User-Agent and refuses to write a manifest naming a URL that
does not answer.

It must also account for what it parses. The name pattern was lowercase-only,
and upstream's ``clin_ct_ALPACA``, ``clin_ct_PUMA``, ``clin_ct_PUMA4`` and
``clin_mr_FVM`` fell through it without a word - 21 of 25 models offered, and
nothing said so. Every ``KEY_URL`` block must now parse as an entry, and a model
that is deliberately not offered is recorded with a reason the generator checks
against the asset. No network here - the HEAD and the archive listing are fakes.
"""
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"

#: The shapes upstream's registry actually has: lowercase names, mixed-case
#: names, an asset with no date in its filename, a trailing comma before the
#: closing brace, and one entry (``clin_mr_FVM``) whose zip does not unpack to
#: the folder the registry names.
REGISTRY = '''
MODEL_METADATA = {
    "clin_ct_body": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_body_27112023.zip",
        KEY_FOLDER_NAME: "Dataset001_body"
    },
    "clin_ct_ALPACA": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_ALPACA.zip",
        KEY_FOLDER_NAME: "Dataset080_Alpaca"
    },
    "clin_ct_PUMA4": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_PUMA4_06032024.zip",
        KEY_FOLDER_NAME: "Dataset003_PUMA4"
    },
    "clin_ct_dental": {
        KEY_URL: "https://model.s.mdforge.com/Dataset112_DentalSegmentator_v100_moose.zip",
        KEY_FOLDER_NAME: "Dataset112_DentalSegmentator_v100"
    },
    "clin_mr_FVM": {
        KEY_URL: "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.2.0/clin_mr_FVM_30032026.zip",
        KEY_FOLDER_NAME: "Dataset501_FVM",
    } 
}

AVAILABLE_MODELS = MODEL_METADATA.keys()


class Model:
    def __init__(self, model_identifier):
        self.url = MODEL_METADATA[model_identifier][KEY_URL]
        if KEY_URL not in MODEL_METADATA[model_identifier]:
            raise ValueError
'''
ALL = {"clin_ct_body", "clin_ct_ALPACA", "clin_ct_PUMA4", "clin_ct_dental", "clin_mr_FVM"}
OFFERED = ALL - {"clin_mr_FVM"}


def fake_top_level(url):
    """What each fake asset unpacks to - FVM's extra parent is the real
    layout of moosez-v.3.2.0's asset, checked 2026-09-06."""
    if "FVM" in url:
        return ["clin_mr_FVM"]
    name = url.rsplit("/", 1)[-1]
    return {"clin_ct_body_27112023.zip": ["Dataset001_body"],
            "clin_ct_ALPACA.zip": ["Dataset080_Alpaca"],
            "clin_ct_PUMA4_06032024.zip": ["Dataset003_PUMA4"],
            "Dataset112_DentalSegmentator_v100_moose.zip": ["Dataset112_DentalSegmentator_v100"]}[name]


@pytest.fixture(scope="module")
def gen():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))       # `import zippeek`, as the script itself does
    spec = importlib.util.spec_from_file_location("gen_moose_manifest", TOOLS / "gen_moose_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def registry(tmp_path):
    p = tmp_path / "models.py"
    p.write_text(REGISTRY)
    return p


def _generate(gen, registry, dest, **kw):
    kw.setdefault("status", lambda url: 200)
    kw.setdefault("top_level", fake_top_level)
    return gen.generate(registry, dest, **kw)


# --- parsing: every entry, whatever its name looks like -----------------------

def test_a_mixed_case_name_is_parsed(gen, registry):
    """The pattern was [a-z0-9_]+ and four upstream models fell through it."""
    entries = gen.parse(registry)
    assert set(entries) == ALL
    assert entries["clin_ct_PUMA4"] == {
        "url": "https://github.com/ENHANCE-PET/MOOSE/releases/download/moosez-v.3.1.3/clin_ct_PUMA4_06032024.zip",
        "folder": "Dataset003_PUMA4", "tag": "06032024"}
    assert entries["clin_mr_FVM"]["folder"] == "Dataset501_FVM"   # trailing comma before }
    assert entries["clin_mr_FVM"]["tag"] == "30032026"
    # no date in the asset name: the release the URL sits under is the tag
    assert entries["clin_ct_ALPACA"]["tag"] == "v.3.1.3"
    # the count is checked against the file, not against what happened to match
    assert len(entries) == REGISTRY.count("KEY_URL:")


@pytest.mark.parametrize("block, why", [
    ('''    "clin_ct_swapped": {
        KEY_FOLDER_NAME: "Dataset009_swapped",
        KEY_URL: "https://h/clin_ct_swapped_01012026.zip"
    },''', "keys in the other order"),
    ('''    "clin-ct-dashed": {
        KEY_URL: "https://h/clin_ct_dashed_01012026.zip",
        KEY_FOLDER_NAME: "Dataset009_dashed"
    },''', "a name character the pattern does not admit"),
    ('''    "clin_ct_const": {
        KEY_URL: SOME_URL,
        KEY_FOLDER_NAME: "Dataset009_const"
    },''', "a value that is not a string literal"),
])
def test_a_block_the_parser_does_not_match_is_refused(gen, tmp_path, block, why):
    """A parser that drops an entry without saying so is the failure the HEAD
    check exists to prevent for URLs. Any KEY_URL block that did not become an
    entry stops the run, and the message names it."""
    p = tmp_path / "models.py"
    p.write_text(REGISTRY.replace('MODEL_METADATA = {\n', 'MODEL_METADATA = {\n' + block + '\n', 1))
    with pytest.raises(SystemExit) as e:
        gen.parse(p)
    msg = str(e.value)
    assert "6 KEY_URL blocks" in msg and "5 parsed" in msg, why
    assert "clin_ct_body" not in msg                    # the ones that parsed are not blamed
    assert ("swapped" in msg or "dashed" in msg or "SOME_URL" in msg), why
    # and generate() never gets as far as writing anything
    with pytest.raises(SystemExit):
        _generate(gen, p, tmp_path / "out.json", not_offered={})
    assert not (tmp_path / "out.json").exists()


def test_a_registry_with_nothing_in_it_is_refused(gen, tmp_path):
    p = tmp_path / "models.py"
    p.write_text("MODELS = {}")
    with pytest.raises(SystemExit, match="no models"):
        _generate(gen, p, tmp_path / "out.json", not_offered={})


# --- the HEAD check --------------------------------------------------------------

def test_a_url_that_does_not_answer_stops_the_manifest(gen, registry, tmp_path):
    dest = tmp_path / "moose_weights.json"
    def status(url):
        return 403 if "mdforge" in url else 200

    with pytest.raises(SystemExit) as e:
        _generate(gen, registry, dest, status=status)
    msg = str(e.value)
    assert "clin_ct_dental" in msg and "403" in msg and "mdforge" in msg
    assert "clin_ct_body" not in msg                 # only the failures are named
    assert "1 of 5" in msg
    assert not dest.exists()


def test_a_failure_below_http_is_reported_by_reason(gen, registry, tmp_path):
    dest = tmp_path / "moose_weights.json"

    def status(url):
        if "mdforge" in url:
            raise urllib.error.URLError("nodename nor servname provided")
        return 200

    with pytest.raises(SystemExit, match="nodename nor servname"):
        _generate(gen, registry, dest, status=status)
    assert not dest.exists()

    def timing_out(url):
        raise TimeoutError("timed out")     # what a socket timeout raises: not a URLError

    with pytest.raises(SystemExit, match="timed out"):
        _generate(gen, registry, dest, status=timing_out)
    assert not dest.exists()


def test_every_url_answering_writes_the_manifest(gen, registry, tmp_path):
    dest = tmp_path / "moose_weights.json"
    asked = []

    def status(url):
        asked.append(url)
        return 200

    tasks = _generate(gen, registry, dest, status=status)
    assert set(tasks) == OFFERED
    assert len(asked) == len(ALL)                   # every recorded URL, the excluded one too, once
    raw = json.loads(dest.read_text())
    assert raw["source"] == "moosez/models.py"
    assert set(raw["tasks"]) == OFFERED
    assert raw["tasks"]["clin_ct_dental"]["tag"] == "v100"
    assert raw["tasks"]["clin_ct_body"]["tag"] == "27112023"
    assert raw["tasks"]["clin_ct_PUMA4"]["folder"] == "Dataset003_PUMA4"


def test_the_check_sends_a_named_user_agent(gen):
    """A HEAD sent as Python-urllib would call the mdforge asset dead; the check
    must identify itself the way the installer does."""
    import urllib.request
    from unittest import mock
    import zippeek
    seen = []

    class R:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(req, *a, **k):
        seen.append(req)
        return R()

    with mock.patch.object(urllib.request, "urlopen", urlopen):
        assert zippeek.head_status("https://h/x.zip") == 200
    assert seen[0].get_method() == "HEAD"
    assert seen[0].get_header("User-agent") == "haversack"


# --- exclusions: recorded with a reason, and the reason is checked ------------------

def test_a_model_not_offered_is_recorded_with_its_reason_checked(gen, registry, tmp_path):
    """clin_mr_FVM's zip unpacks to ``clin_mr_FVM/``, not to the ``Dataset501_FVM``
    the registry names; the installer would refuse it and moosez's own extractor
    would not find it. It is excluded, the manifest says why, and the reason is
    read off the archive rather than taken on trust."""
    dest = tmp_path / "moose_weights.json"
    listed = []

    def top_level(url):
        listed.append(url)
        return fake_top_level(url)

    tasks = _generate(gen, registry, dest, top_level=top_level)
    assert "clin_mr_FVM" not in tasks
    assert listed == [gen.parse(registry)["clin_mr_FVM"]["url"]]   # only the excluded asset is read
    raw = json.loads(dest.read_text())
    assert set(raw["excluded"]) == {"clin_mr_FVM"}
    why = raw["excluded"]["clin_mr_FVM"]
    assert "['clin_mr_FVM']" in why and "'Dataset501_FVM'" in why
    assert "clin_mr_FVM_30032026.zip" in why
    assert not (set(raw["excluded"]) & set(raw["tasks"]))


def test_a_reason_that_has_stopped_being_true_is_an_error(gen, registry, tmp_path):
    """If upstream repackages the asset so it unpacks to Dataset501_FVM, the
    exclusion is stale and must not be carried forward as prose."""
    dest = tmp_path / "moose_weights.json"

    def repackaged(url):
        return ["Dataset501_FVM"] if "FVM" in url else fake_top_level(url)

    with pytest.raises(SystemExit, match="not mispackaged; offer it"):
        _generate(gen, registry, dest, top_level=repackaged)
    assert not dest.exists()


def test_an_exclusion_that_left_the_registry_is_an_error(gen, registry, tmp_path):
    dest = tmp_path / "moose_weights.json"
    with pytest.raises(SystemExit, match="clin_mr_GONE.*not in the registry"):
        _generate(gen, registry, dest, not_offered={"clin_mr_GONE": "mispackaged"})
    assert not dest.exists()


def test_a_reason_the_generator_cannot_check_is_refused(gen, registry, tmp_path):
    dest = tmp_path / "moose_weights.json"
    with pytest.raises(SystemExit, match="not a reason this generator checks"):
        _generate(gen, registry, dest, not_offered={"clin_ct_body": "because"})
    assert not dest.exists()


def test_the_shipped_manifest_has_the_excluded_model_out_of_its_tasks(gen):
    """The catalog reads `tasks` only, so an excluded entry must never also be a task."""
    from haversack.ecosystems import MOOSE_MANIFEST, MooseEcosystem
    raw = json.loads(MOOSE_MANIFEST.read_text())
    assert set(raw["excluded"]) == set(gen.NOT_OFFERED)
    assert not (set(raw["excluded"]) & set(raw["tasks"]))
    offered = MooseEcosystem().tasks()
    assert "clin_mr_FVM" not in offered
    assert {"clin_ct_ALPACA", "clin_ct_PUMA", "clin_ct_PUMA4"} <= set(offered)


def test_top_level_lists_what_the_archive_unpacks_to(gen):
    """The names the installer will compare against the manifest folder - with
    Finder's litter left out, as the installer leaves it out."""
    from unittest import mock
    import zippeek
    listing = {"__MACOSX/._x": 0, "clin_mr_FVM/": 0, "clin_mr_FVM/Dataset501_FVM/": 0,
               "clin_mr_FVM/Dataset501_FVM/t__p__c/dataset.json": 0, ".DS_Store": 0}
    with mock.patch.object(zippeek, "central_directory", lambda url: listing):
        assert zippeek.top_level("https://h/x.zip") == ["clin_mr_FVM"]
