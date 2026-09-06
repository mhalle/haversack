"""Who made each model, under what license, and what its authors ask to be cited.

Three layers, discoverable from any one of them. A *task* has facts of its own
that its catalog manifest carries (a bundle's authors, a per-model license, a
release); its *ecosystem* has the license, the group and the papers; the
*engine* that runs it has a paper of its own (every nnU-Net catalog's authors
ask that nnU-Net be cited alongside their model). :func:`for_task` returns all
three and one ordered ``cite`` list, and is what ``describe()``, every result's
provenance and ``haversack cite`` publish.

The ecosystem and engine facts live in ``data/attribution.json``; every entry
there was read from the project's own README, LICENSE or documentation and
every PMID resolved through PubMed from the DOI the authors publish. Nothing in
this module is written from memory, and a test fails if an ecosystem ships
without a record. Stdlib only, and torch-free: ``describe()`` runs on the lean
API image.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data" / "attribution.json"

_DOI = re.compile(r"10\.\d{4,9}/[^\s\"'<>)]+")
_ARXIV = re.compile(r"arXiv[: ]\s*(\d{4}\.\d{4,5})", re.I)


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(DATA.read_text())


def for_ecosystem(name: str) -> dict | None:
    rec = load()["ecosystems"].get(name)
    return dict(rec) if rec else None


def for_engine(name: str) -> dict | None:
    rec = load()["engines"].get(name)
    if rec and rec.get("same_as_ecosystem"):
        return for_ecosystem(rec["same_as_ecosystem"])
    return dict(rec) if rec else None


def _applies(ref: dict, info: dict) -> bool:
    """A reference marked ``when: {modality: MR}`` is asked for only by the
    models of that modality (TotalSegmentator's MRI paper for its MR tasks)."""
    when = ref.get("when") or {}
    modality = str(info.get("modality") or "").upper()
    return all(modality.startswith(str(v).upper()) for k, v in when.items() if k == "modality")


def _parse_reference(text: str) -> dict:
    """A free-text reference (a MONAI bundle's) with whatever identifiers it
    carries lifted out, so it can be deduplicated against structured ones."""
    ref = {"text": text.strip()}
    doi = _DOI.search(text)
    if doi:
        ref["doi"] = doi.group(0).rstrip(".")
    arx = _ARXIV.search(text)
    if arx:
        ref["arxiv"] = arx.group(1)
    return ref


def _key(ref: dict) -> str:
    return (ref.get("doi") or ref.get("arxiv") or ref.get("pmid")
            or ref.get("title") or ref.get("text") or "").lower()


def _task_block(short: str, eco: dict, info: dict) -> dict:
    """The task's own facts: what its manifest recorded, plus the per-task
    license rules an ecosystem states (TotalSegmentator's licensed models)."""
    task = {}
    for key in ("description", "summary", "release", "tag", "authors", "copyright",
                "data_source", "bundle_version"):
        if info.get(key):
            task[key] = info[key]
    lic = info.get("license")
    if isinstance(lic, str):
        task["license"] = {"weights": lic}
    elif isinstance(lic, dict):
        task["license"] = dict(lic)
    if short in (eco.get("licensed_tasks") or ()):
        task["license"] = {
            "weights": "TotalSegmentator model license: free for non-commercial use, "
                       "paid for commercial use, obtained from the authors",
            "url": (eco.get("license") or {}).get("url"),
            "note": "haversack does not handle this license and will not download these "
                    "weights; install them through TotalSegmentator's own flow"}
    if eco.get("title") == "TotalSegmentator" and short == "brain_aneurysm":
        task["license"] = {"weights": "CC-BY-NC-4.0",
                           "note": "non-commercial only; no commercial license is offered"}
    refs = [_parse_reference(r) for r in (info.get("references") or []) if str(r).strip()]
    if refs:
        task["references"] = refs
    return task


def for_task(canonical: str, info: dict | None = None) -> dict:
    """Everything needed to credit one task, from any of its three layers.

    ``info`` is the catalog's ``info()`` record when the caller has one: it
    supplies the ecosystem and engine names, the modality (which decides
    whether a modality-specific paper applies) and the manifest's own facts.
    Without it the ecosystem is read off the ``eco:task`` grammar and the engine
    defaults to nnU-Net.
    """
    info = dict(info or {})
    eco_name = info.get("ecosystem") or (canonical.partition(":")[0] if ":" in canonical else "")
    short = canonical.partition(":")[2] if ":" in canonical else canonical
    eco = for_ecosystem(eco_name) or {}
    eng_name = info.get("engine") or "nnunetv2"
    eng = for_engine(eng_name) or {}
    task = _task_block(short, eco, info)
    # A task whose model comes from somewhere other than its catalog (MOOSE
    # redistributes DentalSegmentator's checkpoint) credits its real makers:
    # their license governs the output, and their paper leads the list.
    override = (load().get("tasks") or {}).get(canonical) or {}
    origin = for_ecosystem(override.get("derived_from", "")) if override.get("derived_from") else None
    if origin:
        task["derived_from"] = override["derived_from"]
        task["origin"] = {k: origin[k] for k in ("title", "group", "repository", "license") if k in origin}
        if origin.get("license"):
            task["license"] = dict(origin["license"])
        if override.get("note"):
            task["note"] = override["note"]

    cite, seen = [], set()

    def add(ref: dict, source: str):
        k = _key(ref)
        if k and k in seen:
            return
        seen.add(k)
        cite.append({**ref, "for": source})

    for ref in task.get("references") or []:
        add(ref, "task")
    for ref in (origin or {}).get("cite") or []:
        add({k: v for k, v in ref.items() if k != "when"}, "task")
    for ref in eco.get("cite") or []:
        if _applies(ref, info):
            add({k: v for k, v in ref.items() if k != "when"}, "ecosystem")
    for also in eco.get("also_cite") or []:
        for ref in (for_engine(also) or {}).get("cite") or []:
            add(ref, "engine")
    if eng is not eco:
        for ref in eng.get("cite") or []:
            add(ref, "engine")

    def summary(rec: dict) -> dict:
        return {k: rec[k] for k in ("title", "description", "group", "repository", "url",
                                    "license") if k in rec}

    return {"task": task,
            "ecosystem": eco_name, "ecosystem_info": summary(eco),
            "engine": eng_name, "engine_info": summary(eng),
            "cite": cite}


def provenance_block(canonical: str, info: dict | None = None) -> dict:
    """The compact form written into every result: the license that governs
    the output, and the identifiers of what to cite. Identifiers only - the
    full record is a ``describe()`` away, and a seg.nrrd header is not a
    bibliography."""
    rec = for_task(canonical, info)
    lic = rec["task"].get("license")
    if not lic:
        # the ecosystem's names only: its note about OTHER models' terms
        # belongs in describe(), not in the header of this result
        eco_lic = rec["ecosystem_info"].get("license")
        lic = ({k: v for k, v in eco_lic.items() if k in ("code", "weights") and v}
               if isinstance(eco_lic, dict) else eco_lic)
    return {"ecosystem": rec["ecosystem"], "engine": rec["engine"],
            "license": lic,
            "cite": [{k: r[k] for k in ("doi", "pmid", "arxiv", "title") if k in r}
                     for r in rec["cite"]]}


def format_reference(ref: dict) -> str:
    if ref.get("text") and not ref.get("title"):
        line = ref["text"]
    else:
        bits = [ref.get("authors"), ref.get("title"),
                " ".join(str(x) for x in (ref.get("journal"), ref.get("year")) if x)]
        line = ". ".join(str(b).rstrip(".") for b in bits if b) + "."
    ids = []
    if ref.get("doi") and ref["doi"] not in line:
        ids.append(f"doi:{ref['doi']}")
    if ref.get("pmid"):
        ids.append(f"PMID {ref['pmid']}")
    if ref.get("arxiv") and ref["arxiv"] not in line:
        ids.append(f"arXiv:{ref['arxiv']}")
    if ref.get("note"):
        ids.append(f"({ref['note']})")
    return line + (("  " + "  ".join(ids)) if ids else "")


def _license_line(lic) -> str:
    if not lic:
        return "not stated"
    if isinstance(lic, str):
        return lic
    parts = [f"{k} {v}" for k, v in lic.items() if k in ("code", "weights") and v]
    if lic.get("url"):
        parts.append(lic["url"])
    line = "; ".join(parts) or "not stated"
    if lic.get("note"):
        line += f"\n      {lic['note']}"
    return line


def format(canonical: str, info: dict | None = None) -> str:
    """What ``haversack cite <task>`` prints: the three layers, then the list."""
    rec = for_task(canonical, info)
    eco, eng, task = rec["ecosystem_info"], rec["engine_info"], rec["task"]
    out = [f"{canonical}"]
    if eco:
        out.append(f"  ecosystem {rec['ecosystem']}: {eco.get('title', '')}")
        if eco.get("description"):
            out.append(f"    {eco['description']}")
        if eco.get("group"):
            out.append(f"    group:      {eco['group']}")
        if eco.get("repository"):
            out.append(f"    repository: {eco['repository']}")
        out.append(f"    license:    {_license_line(eco.get('license'))}")
    if task:
        out.append("  this task:")
        for key in ("description", "summary", "release", "authors", "copyright", "data_source"):
            if task.get(key):
                out.append(f"    {key + ':':<13}{task[key]}")
        if task.get("origin"):
            o = task["origin"]
            out.append(f"    made by:     {o.get('title', '')} - {o.get('group', '')}")
            if o.get("repository"):
                out.append(f"                 {o['repository']}")
        if task.get("note"):
            out.append(f"    note:        {task['note']}")
        if task.get("license"):
            out.append(f"    license:    {_license_line(task['license'])}")
    if eng and rec["engine"] != rec["ecosystem"]:
        out.append(f"  engine {rec['engine']}: {eng.get('title', '')}"
                   + (f" ({eng['license'].get('code')})" if isinstance(eng.get("license"), dict) else ""))
    if rec["cite"]:
        out.append("  please cite:")
        for i, ref in enumerate(rec["cite"], 1):
            out.append(f"    {i}. {format_reference(ref)}")
    else:
        out.append("  no citation is recorded for this task")
    return "\n".join(out)
