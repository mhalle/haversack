"""The house spelling is American, in every tracked text file (2026-09-22).

Asked for on 2026-09-06 and again, as critical, on 2026-09-22, when new scripts had picked up
"centre" from a sibling repository's API. The reason is DRIFT: every British word already in a
file is a template the next edit copies, so a clean tree has to be held clean by a test, not by
care. Tests and tools are scanned too - they are not user facing, but they are where the copying
starts.

The detector is an explicit list of British forms, never a suffix rule: "-ise" would flag advise,
exercise, precise and noise. Identifiers are split before matching (``centre_err``, ``colourMap``,
``NormaliseHU``), so a local name drifts no more quietly than a comment does.

Exceptions have to be named where they are and have to stay true:

- ``cancelled`` / ``Cancelled`` are a job state on the wire and an exception class; respelling
  either breaks clients (``GLOBAL_OK``, and a test that the wire state still exists).
- A line that must quote someone else's spelling (a third-party attribute, a paper's title) says
  so on that line: ``spelling: allow <word>``. It allows that word on that line only, and a pragma
  whose word is no longer on its line fails the test - an exception may not outlive its reason.
- This file lists the British forms it looks for, so it is the one file not scanned; the detector
  tests below are what hold it to its job.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SELF = pathlib.Path(__file__).resolve().relative_to(ROOT).as_posix()
TEXT = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".html", ".txt", ".cfg", ".ini", ".sh",
        ".js", ".mjs", ".cjs", ".ts", ".tsx"}
NOT_SCANNED = {SELF: "lists the British forms it detects"}

# -- the British forms, generated from stems -----------------------------------------------------
_OUR = ("colour behaviour favour honour neighbour labour humour flavour harbour rumour vapour odour "
        "tumour armour endeavour savour vigour rigour valour splendour parlour").split()
_OUR_SUFFIX = ("", "s", "ed", "ing", "ful", "less", "hood", "hoods", "al", "ally", "ite", "ites", "able",
               "ably", "map", "maps", "bar", "bars", "ation", "ise", "ised", "ize", "ized", "er", "ers")
_RE = ("centre metre litre fibre calibre theatre spectre sombre lustre meagre sabre ochre "
       "millimetre centimetre kilometre micrometre nanometre millilitre epicentre").split()
_RE_SUFFIX = ("", "s", "d", "line", "lines", "point", "points")
_ISE = ("normalis organis recognis optimis minimis maximis visualis summaris generalis initialis serialis "
        "deserialis parallelis prioritis categoris characteris finalis standardis synchronis customis authoris "
        "utilis emphasis realis specialis materialis tokenis quantis discretis binaris randomis regularis "
        "penalis vectoris rasteris digitis memoris apologis criticis localis centralis stabilis neutralis "
        "capitalis canonicalis mobilis polaris symbolis harmonis idealis sanitis anonymis pseudonymis "
        "reorganis reinitialis uninitialis unrecognis").split()
_ISE_SUFFIX = ("e", "es", "ed", "ing", "er", "ers", "ation", "ations", "able")
_YSE = {"analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing", "analyser": "analyzer",
        "analysers": "analyzers", "paralyse": "paralyze", "paralysed": "paralyzed", "catalyse": "catalyze",
        "catalysed": "catalyzed", "catalysing": "catalyzing"}          # never "analyses": a US plural too
_LL = ("unlabelled relabelled relabelling unmodelled "
       "labelled labelling modelled modelling modeller modellers travelled travelling traveller travellers "
       "signalled signalling levelled levelling fuelled fuelling channelled channelling tunnelled funnelled "
       "totalled totalling dialled dialling counselled marshalled marshalling cancelled cancelling "
       "jewellery woollen").split()
_OTHER = {
    "licence": "license", "licences": "licenses", "grey": "gray", "greys": "grays", "greyscale": "grayscale",
    "whilst": "while", "amongst": "among", "artefact": "artifact", "artefacts": "artifacts",
    "programme": "program", "programmes": "programs", "catalogue": "catalog", "catalogues": "catalogs",
    "catalogued": "cataloged", "analogue": "analog",
    "defence": "defense", "offence": "offense", "pretence": "pretense",
    "judgement": "judgment", "judgements": "judgments", "aluminium": "aluminum", "sceptical": "skeptical",
    "manoeuvre": "maneuver", "manoeuvres": "maneuvers", "learnt": "learned", "spelt": "spelled",
    "fulfil": "fulfill", "fulfilment": "fulfillment", "enrol": "enroll", "enrolment": "enrollment",
    "instalment": "installment", "skilful": "skillful", "wilful": "willful", "centring": "centering",
    # the medical forms this domain meets
    "oedema": "edema", "oesophagus": "esophagus", "oesophageal": "esophageal", "haemorrhage": "hemorrhage",
    "haemorrhagic": "hemorrhagic", "haematoma": "hematoma", "haemoglobin": "hemoglobin",
    "haematocrit": "hematocrit", "anaemia": "anemia", "ischaemia": "ischemia", "ischaemic": "ischemic",
    "leukaemia": "leukemia", "paediatric": "pediatric", "paediatrics": "pediatrics", "anaesthesia": "anesthesia",
    "anaesthetic": "anesthetic", "foetal": "fetal", "foetus": "fetus", "diarrhoea": "diarrhea",
    "oestrogen": "estrogen", "orthopaedic": "orthopedic", "caesarean": "cesarean", "gynaecology": "gynecology",
    "coeliac": "celiac",
}


def _forms() -> dict[str, str]:
    out = {}
    for s in _OUR:
        for x in _OUR_SUFFIX:
            out[s + x] = s.replace("our", "or") + x.replace("ise", "ize")
    for s in _RE:
        for x in _RE_SUFFIX:
            out[s + x] = s[:-2] + "er" + ("ed" if x == "d" else x)
    for s in _ISE:
        for x in _ISE_SUFFIX:
            out[s + x] = s[:-1] + "z" + x
    out.pop("emphasis", None)                                  # the noun is the same in both
    for w in _LL:
        i = w.rindex("ll")
        out[w] = w[:i] + w[i + 1:]
    out.update(_YSE)
    out.update(_OTHER)
    return out


BRITISH = _forms()

# A job state on the wire (jobpolicy.TERMINAL, every client) and the Cancelled exception class.
GLOBAL_OK = {"cancelled": "the job state on the wire and the Cancelled class", "cancelling": "the same family"}

_WORD = re.compile(r"[A-Za-z]+")
_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")
_PRAGMA = re.compile(r"spelling:\s*allow\s+([A-Za-z ,]+)")


def tokens(line: str):
    """Every word of a line, identifiers split at underscores, digits and camelCase, lower-cased."""
    for w in _WORD.findall(line):
        yield w.lower()
        parts = _PART.findall(w)
        if len(parts) > 1:
            for p in parts:
                yield p.lower()


def british_in(line: str) -> list[tuple[str, str]]:
    seen, out = set(), []
    for t in tokens(line):
        if t in BRITISH and t not in GLOBAL_OK and t not in seen:
            seen.add(t); out.append((t, BRITISH[t]))
    return out


def tracked_text_files() -> list[str]:
    r = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if r.returncode:
        raise AssertionError("the spelling test reads the tracked files from git: run it in a git checkout "
                             f"({r.stderr.decode(errors='replace').strip()})")
    names = [n for n in r.stdout.decode().split("\0") if n]
    return sorted(n for n in names if pathlib.PurePosixPath(n).suffix in TEXT and n not in NOT_SCANNED)


def scan(names) -> tuple[list[str], list[str]]:
    """``(violations, stale pragmas)`` as ``path:line: ...`` lines."""
    bad, stale = [], []
    for n in names:
        try:
            text = (ROOT / n).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = _PRAGMA.search(line)
            allowed = {w.lower() for w in re.split(r"[ ,]+", m.group(1)) if w} if m else set()
            body = line[:m.start()] + line[m.end():] if m else line      # the pragma names its word: not evidence
            found = british_in(body)
            for w in sorted(allowed - {t for t, _ in found}):
                stale.append(f"{n}:{i}: 'spelling: allow {w}' but {w!r} is not on this line - remove the pragma")
            for t, us in found:
                if t not in allowed:
                    bad.append(f"{n}:{i}: {t!r} -> {us!r}")
    return bad, stale


class TestAmericanSpelling(unittest.TestCase):
    def test_no_british_spelling_in_tracked_text(self):
        bad, stale = scan(tracked_text_files())
        msg = ("house spelling is American (drift: every British word is a template the next edit copies). "
               "Quoting someone else's name? put 'spelling: allow <word>' on that line.\n")
        self.assertEqual(bad, [], msg + "\n".join(bad))

    def test_every_pragma_still_has_its_reason(self):
        _, stale = scan(tracked_text_files())
        self.assertEqual(stale, [], "\n".join(stale))


class TestTheScanSeesTheRepo(unittest.TestCase):
    """A file filter that matched nothing would pass the test above trivially."""

    def test_the_scan_covers_the_package_the_docs_and_the_tests(self):
        names = set(tracked_text_files())
        self.assertGreater(len(names), 150)
        for must in ("src/haversack/serve.py", "README.md", "SERVER.md", "CHANGELOG.md", "pyproject.toml",
                     "tests/test_serve.py"):
            self.assertIn(must, names)
        self.assertNotIn(SELF, names)

    def test_the_global_exception_is_still_the_wire_state(self):
        """If the job state is ever respelled, ``cancelled`` stops being an exception."""
        src = (ROOT / "src/haversack/jobpolicy.py").read_text()
        self.assertRegex(src, r'TERMINAL = \([^)]*"cancelled"')


class TestTheDetector(unittest.TestCase):
    def test_catches_prose_and_identifiers(self):
        for line, word in (("the centre of the box", "centre"), ("colourMap = cm.viridis", "colour"),
                           ("centre_err = lo - c", "centre"), ("x = NormaliseHU(ct)", "normalise"),
                           ("voxels labelled 3", "labelled"), ("a tumour of 2 ml", "tumour"),
                           ("in millimetres", "millimetres"), ("its behaviour under load", "behaviour"),
                           ("we analysed it", "analysed"), ("whilst holding the lock", "whilst"),
                           ("CC BY licence", "licence"), ("the oesophagus", "oesophagus"),
                           ("neighbourhood of radius r", "neighbourhood"), ("summarising", "summarising"),
                           ("voxels left unlabelled", "unlabelled"), ("an analogue of it", "analogue"),
                           ("pass  # store uninitialised", "uninitialised")):
            self.assertIn(word, [t for t, _ in british_in(line)], line)

    def test_spares_american_and_shared_words(self):
        for line in ("the center of the box", "colorMap", "canceled", "job state cancelled", "raise Cancelled()",
                     "advise exercise precise noise otherwise premise expertise compromise surprise",
                     "parameter diameter perimeter", "two analyses of emphasis", "organism organization",
                     "gray labeled modeling tumor neighbor behavior license analyze normalize"):
            self.assertEqual(british_in(line), [], line)

    def test_a_pragma_allows_only_its_word_on_its_line(self):
        import tempfile
        global ROOT
        old = ROOT
        with tempfile.TemporaryDirectory() as d:
            ROOT = pathlib.Path(d)
            (ROOT / "x.py").write_text("c = sw.centre  # spelling: allow centre\n"
                                       "colour = 1  # spelling: allow centre\n"
                                       "ok = 2  # spelling: allow grey\n")
            try:
                bad, stale = scan(["x.py"])
            finally:
                ROOT = old
        self.assertEqual(bad, ["x.py:2: 'colour' -> 'color'"])
        self.assertEqual(len(stale), 2)                        # line 2's centre and line 3's grey are not there


if __name__ == "__main__":
    unittest.main()
