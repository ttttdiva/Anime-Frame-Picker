"""Offline additive adoption of pinned accepted reports and *existing* caches.

No batch/model/video imports. Candidate PNG decoding reproduces the tag cache
key only. The sole copied media are the accepted overlay additions. Publishing
current.json is deliberately a separate action after API and real UI checks.
"""
from __future__ import annotations

import argparse
import array
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anime_picker.accepted_catalogue import (SCHEMA, RELEASE_RE, canonical, create_schema,
    digest, normalized, pack_scores, png_dimensions, safe_file, signature, stable_id)


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")


def pinned(ref: dict, pins: dict):
    path = Path(ref["path"])
    actual = digest(path)
    if actual != ref["sha256"]:
        raise ValueError(f"pinned artifact mismatch: {path}")
    previous = pins.setdefault(str(path), actual)
    if previous != actual:
        raise ValueError(f"conflicting artifact pins: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metadata(path: Path):
    s = path.lstat()
    return {k: getattr(s, k) for k in ("st_mode", "st_size", "st_ino", "st_dev", "st_nlink", "st_mtime_ns")}


def join_records(accepted: list, old: dict, mapping: list):
    """Exact raw ID and integer PTS joins; never deduplicate by PTS alone."""
    def identity(record):
        if type(record["pts"]) is not int or not isinstance(record["id"], str):
            raise ValueError("noninteger PTS or nonstring input ID")
        return (record["id"], record["pts"])
    source = {}
    for record in [*old["kept"], *old["excluded"]]:
        key = identity(record)
        if key in source:
            raise ValueError("duplicate old input identity")
        source[key] = record
    native = {}
    for record in mapping:
        key = identity(record)
        if key in native:
            raise ValueError("duplicate native mapping identity")
        native[key] = record
    seen = set()
    for record in sorted(accepted, key=lambda r: (r["pts"], r["id"])):
        key = identity(record)
        if key in seen or key not in source or key not in native or not native[key]["new_kept"]:
            raise ValueError("accepted set/native/input identity mismatch")
        seen.add(key)
        prior = source[key]
        if record["image"] != prior["image"] or record.get("detections") != prior.get("detections"):
            raise ValueError("accepted input image/detections drift")
        yield record, prior, native[key]
    if seen != {identity(r) for r in mapping if r["new_kept"]}:
        raise ValueError("accepted/native set mismatch")


def full_tags(record, old, fingerprint, revision):
    """Read existing cache JSON only; the production callsite supplies its root.

    batch_source._curate_character_candidates receives args.cache_root/source_hash;
    its pinned detector_backend.cache_directory is that exact root/detections/rev.
    Detector records bind it to the source fingerprint and candidate PNG digest.
    """
    image = Path(record["image"])
    payload = image.read_bytes()
    image_sha = hashlib.sha256(payload).hexdigest()
    detector_root = Path(old["detector_backend"]["cache_directory"])
    detector_path = detector_root / f'pts_{record["pts"]}.json'
    detector_raw = detector_path.read_bytes()
    detector = json.loads(detector_raw)
    if (detector["source_fingerprint"] != fingerprint or detector["pts"] != record["pts"]
            or detector["image_sha256"] != image_sha or detector["detections"] != record["detections"]
            or detector["detector_revision"] != old["detector_backend"]["cache_revision"]):
        raise ValueError("candidate/source/detector provenance mismatch")
    from PIL import Image  # CPU PNG reader only, never a model service.
    with Image.open(io.BytesIO(payload)) as im:
        rgb_key = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
    cache_root = detector_root.parent.parent
    if cache_root.name != fingerprint.rsplit(":", 1)[-1][:12]:
        raise ValueError("pinned per-source cache root mismatch")
    revision_dir = hashlib.sha256(revision.encode("utf-8")).hexdigest()[:16]
    cache_path = cache_root / "tags" / revision_dir / (rgb_key + ".json")
    evidence = {"status": "missing", "input_image": str(image), "input_sha256": image_sha,
        "rgb_key": rgb_key, "tag_revision": revision, "cache_root": str(cache_root),
        "cache_path": str(cache_path), "detector_cache_path": str(detector_path),
        "detector_cache_sha256": hashlib.sha256(detector_raw).hexdigest(),
        "record_identity": {"id": record["id"], "pts": record["pts"], "source_fingerprint": fingerprint}}
    try:
        raw = cache_path.read_bytes()
        evidence["cache_sha256"] = hashlib.sha256(raw).hexdigest()
        tags = json.loads(raw)
        if not isinstance(tags, dict) or not tags or "character" not in tags:
            raise ValueError("full tag categories absent")
        for category, values in tags.items():
            if not isinstance(category, str) or not isinstance(values, dict):
                raise ValueError("invalid tag category")
            for name, score in values.items():
                if (not isinstance(name, str) or isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("invalid full tag score")
        evidence["status"] = "complete"
        evidence["cache_bytes"] = len(raw)
        return tags, evidence
    except (OSError, ValueError, TypeError) as exc:
        evidence["reason"] = type(exc).__name__ + ": " + str(exc)
        return {}, evidence


def build(evidence: Path, release_id: str, output_root: Path, resume=False):
    started, cpu = time.perf_counter(), time.process_time()
    journal = json.loads((evidence / "journal.json").read_text(encoding="utf-8"))
    if not RELEASE_RE.fullmatch(release_id):
        raise ValueError("invalid release ID")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != journal["head"]:
        raise ValueError("source HEAD drift")
    if digest(ROOT / ".git/index") != journal["index_sha256"]:
        raise ValueError("Git index drift")
    pins = {}
    documents = {Path(path).name: pinned({"path": path, "sha256": sha}, pins)
                 for path, sha in journal["input_pins"].items()}
    aggregate = documents["aggregate.json"]
    overlay = documents["accepted-overlay-manifest.json"]
    interpretation = documents["acceptance-interpretation.json"]
    if (interpretation["status"] != "PASS_PRIVATE_NATIVE_CLOSURE_WITH_EXACT_DIRECTORY_SIZE_EXCEPTION"
            or aggregate["provenance_mode"] != "MIXED_SOURCE_ACCEPTED_REUSE"
            or overlay["totals"]["accepted_frames"] != 23792):
        raise ValueError("accepted closure status/count mismatch")
    # Retain and recheck nested historical pins; do not normalize old receipts.
    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                p = Path(value["path"])
                if p.is_file() and len(value["sha256"]) == 64:
                    actual = digest(p)
                    if actual != value["sha256"]:
                        raise ValueError(f"historical pin drift: {p}")
                    pins[str(p)] = actual
            for v in value.values():
                collect(v)
        elif isinstance(value, list):
            for v in value:
                collect(v)
    for doc in documents.values():
        collect(doc)
    specs = [(r["key"], int(r["key"]), r["report"], r["native_manifest"], r["source"], r["new_kept"]) for r in aggregate["rows"]]
    ep = aggregate["accepted_ep1"]
    specs.append(("accepted_ep1", 23, {"path": ep["report_path"], "sha256": ep["report_sha256"]}, ep["native_mapping"], ep["source"], ep["counts"]["kept"]))
    specs.sort(key=lambda s: s[1])
    additions = {(a["row_key"], a["id"], a["pts"]): a for a in overlay["additions"]}
    if len(additions) != 23 or len(specs) != 23:
        raise ValueError("wrong overlay/row cardinality")
    release = output_root / "_accepted" / release_id
    if resume:
        if not release.is_dir() or (release / "manifest.json").exists():
            raise ValueError("only an unfinished exact release can resume")
    else:
        release.mkdir(parents=True, exist_ok=False)
        write_new(evidence / "build-plan.json", {"release": str(release), "source_pins": pins,
            "specs": specs, "builder_sha256": digest(Path(__file__)), "source_hashes": {
                p: digest(ROOT / p) for p in journal["tracked_hashes"] if (ROOT / p).is_file()},
            "max_score_buffer_bytes_estimate": max(s[5] for s in specs) * 30877 * 8,
            "estimated_uncompressed_score_bytes": 23792 * 30877 * 8,
            "cache_key_recipe": "Pillow RGB contiguous uint8 bytes == batch_source numpy RGB bytes",
            "pointer_baseline": journal["pointer_baseline"]})
    db = sqlite3.connect(release / "index.sqlite")
    try:
        if not resume:
            create_schema(db)
            db.execute("CREATE TABLE build_rows(row_key TEXT PRIMARY KEY, receipt TEXT NOT NULL)")
            db.commit()
        vocab = {(r[1], r[2]): r[0] for r in db.execute("SELECT tag_id,category,name FROM vocabulary")}
        receipts, rows = [], []
        seq = 0
        for key, order, report_ref, native_ref, source_version, expected_count in specs:
            saved = db.execute("SELECT receipt FROM build_rows WHERE row_key=?", (key,)).fetchone()
            if saved:
                receipt = json.loads(saved[0])
                # Revalidate every pinned native, input/cache and row artifact before skipping.
                for path, info in receipt["files"].items():
                    p = Path(path)
                    if metadata(p) != info["metadata"] or digest(p) != info["sha256"]:
                        raise ValueError(f"resumed row validation drift: {p}")
                receipts.append(receipt)
                rows.append(receipt["row"])
                if not (evidence / f"row-{key}.json").exists():
                    write_new(evidence / f"row-{key}.json", receipt)
                seq += receipt["count"]
                print(canonical({"row": key, "status": "validated_resume", "count": receipt["count"]}), flush=True)
                continue
            row_wall, row_cpu = time.perf_counter(), time.process_time()
            files = {}
            def capture(path, sha=None):
                path = Path(path)
                files[str(path)] = {"metadata": metadata(path), "sha256": sha or digest(path)}
            wrapper = pinned(report_ref, pins)
            accepted = wrapper["report"]
            mapping = pinned(native_ref, pins)
            batch_ref = mapping["batch_result"]
            batch = pinned(batch_ref, pins)
            old_path = Path(batch["analysis"]["curation"]["report"])
            old_hash = next(r["report_sha256"] for r in aggregate["old_reports_23"]
                            if r["key"] == key)
            old = pinned({"path": str(old_path), "sha256": old_hash}, pins)
            revision = old["tagger_revision"]
            fingerprint = batch["fingerprint"]
            if revision != batch["model_revision"] or revision != old["tagging"]["revision"]:
                raise ValueError("tag model revision mismatch")
            meta_path = Path(batch_ref["path"]).parent / "character_01/metadata.json"
            native_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if native_metadata["source_fingerprint"] != fingerprint or native_metadata["model_revision"] != revision:
                raise ValueError("native source fingerprint/revision mismatch")
            capture(meta_path)
            for ref in (report_ref, native_ref, batch_ref, {"path": str(old_path), "sha256": old_hash}):
                capture(ref["path"], ref["sha256"])
            if len(accepted["kept"]) != expected_count:
                raise ValueError("per-row accepted count mismatch")
            columns = {}
            first_seq = seq
            missing, copies, scores, cache_bytes = 0, 0, 0, 0
            suggestions = Counter()
            excluded = [(r["id"], r["pts"]) for r in mapping["rows"] if r["old_kept"] and not r["new_kept"]]
            overlay_ids = []
            for index, (record, prior, mapped) in enumerate(join_records(accepted["kept"], old["report"], mapping["rows"])):
                identity = (key, record["id"], record["pts"])
                add = additions.get(identity)
                if add:
                    if (add["accepted_report_source"] != source_version
                            or any(add["native_manifest"][k] != native_ref[k] for k in ("path", "sha256"))
                            or any(add["source_batch_result"][k] != batch_ref[k] for k in ("path", "sha256"))):
                        raise ValueError("overlay native/batch/source-version binding mismatch")
                    if mapped["native_status"] != "missing_native" or add["fingerprint"] != fingerprint or add["accepted_report"] != report_ref:
                        # References can include extra readback fields; compare authoritative fields below.
                        if (mapped["native_status"] != "missing_native" or add["fingerprint"] != fingerprint
                                or any(add["accepted_report"][k] != report_ref[k] for k in ("path", "sha256"))):
                            raise ValueError("overlay report/source binding mismatch")
                    relative = f"_accepted/{release_id}/native/row-{key}/pts-{record['pts']}.png"
                    destination = output_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    original = Path(add["path"])
                    payload = original.read_bytes()
                    native_hash = hashlib.sha256(payload).hexdigest()
                    width, height = png_dimensions(payload)
                    if native_hash != add["sha256"] or (width, height) != (add["width"], add["height"]):
                        raise ValueError("overlay PNG mismatch")
                    capture(original, native_hash)
                    if destination.exists():
                        if not resume or digest(destination) != native_hash:
                            raise ValueError("overlay destination collision")
                    else:
                        with destination.open("xb") as stream:
                            stream.write(payload)
                    if digest(destination) != native_hash or destination.stat().st_nlink != 1:
                        raise ValueError("copied overlay verification failed")
                    from PIL import Image
                    with Image.open(destination) as im:
                        im.load()
                        if im.format != "PNG" or im.size != (width, height):
                            raise ValueError("copied native dimensions mismatch")
                    capture(destination, native_hash)
                    copies += 1
                    origin = "accepted_overlay_copy"
                    overlay_ids.append(identity)
                else:
                    native = mapped["native"]
                    output = native["output_entry"]
                    relative = output.get("public_file") or native["public_entry"]["file"]
                    image = safe_file(output_root, relative, ".png")
                    native_hash = output.get("public_sha256") or output["sha256"]
                    payload_hash = digest(image)
                    with image.open("rb") as stream:
                        width, height = png_dimensions(stream.read(33))
                    location = next(l for l in native["locations"] if l["kind"] == "public")
                    if payload_hash != native_hash or (width, height) != (location["width"], location["height"]):
                        raise ValueError("existing public native hash/dimensions mismatch")
                    capture(image, native_hash)
                    origin = "existing_native"
                tags, tag_evidence = full_tags(prior, old, fingerprint, revision)
                tag_evidence["record_identity"]["row_key"] = key
                capture(prior["image"], tag_evidence["input_sha256"])
                capture(tag_evidence["detector_cache_path"], tag_evidence["detector_cache_sha256"])
                if tag_evidence["status"] == "complete":
                    capture(tag_evidence["cache_path"], tag_evidence["cache_sha256"])
                    cache_bytes += tag_evidence["cache_bytes"]
                else:
                    missing += 1
                for category, values in tags.items():
                    for name, score in values.items():
                        tag_id = vocab.get((category, name))
                        if tag_id is None:
                            tag_id = len(vocab) + 1
                            db.execute("INSERT INTO vocabulary VALUES(?,?,?,?)", (tag_id, category, name, normalized(name)))
                            vocab[category, name] = tag_id
                        if tag_id not in columns:
                            columns[tag_id] = array.array("d", [math.nan]) * expected_count
                        columns[tag_id][index] = score
                        scores += 1
                        if category == "character" and score >= 0.1:
                            suggestions[name] += 1
                entry_id = stable_id(key, record["id"], record["pts"], fingerprint)
                entry = {"row_key": key, "raw_id": record["id"], "pts": record["pts"],
                    "source_fingerprint": fingerprint, "source_name": Path(batch["video"]).name,
                    "time_base": native_metadata["source_timebase"], "time": record["time"],
                    "accepted_report": report_ref, "accepted_source_version": source_version,
                    "native_mapping": native_ref, "source_batch_result": batch_ref,
                    "old_input_report": {"path": str(old_path), "sha256": old_hash},
                    "native_origin": origin, "overlay_provenance": add,
                    "accepted_record": record, "tag_evidence": tag_evidence,
                    "group_evidence": {"report": report_ref, "group_id": record.get("group_id")}}
                db.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?,?,?)", (seq, entry_id, key, order,
                    record["pts"], record["id"], relative, native_hash, width, height, canonical(entry)))
                seq += 1
                if (index + 1) % 100 == 0:
                    print(canonical({"row": key, "validated": index + 1, "of": expected_count}), flush=True)
            for tag_id in sorted(columns):
                db.execute("INSERT INTO scores VALUES(?,?,?,?,?)", (tag_id, key, first_seq, expected_count, pack_scores(columns[tag_id])))
            del columns
            row_summary = {"key": key, "order": order, "count": expected_count,
                           "name": Path(batch["video"]).stem, "missing_tags": missing}
            receipt = {"row": row_summary, "count": expected_count, "overlay_copies": copies,
                "overlay_identities": overlay_ids, "excluded_old_kept": excluded,
                "report": report_ref, "native_mapping": native_ref, "source_version": source_version,
                "batch": batch_ref, "source_video": batch["video"], "source_fingerprint": fingerprint,
                "cache_bytes_read": cache_bytes, "scores_preserved": scores,
                "candidate_png_decodes": expected_count, "missing_tags": missing,
                "suggestions": dict(suggestions), "files": files,
                "wall_seconds": time.perf_counter() - row_wall, "cpu_seconds": time.process_time() - row_cpu}
            db.execute("INSERT INTO build_rows VALUES(?,?)", (key, canonical(receipt)))
            db.commit()
            row_file = evidence / f"row-{key}.json"
            if not row_file.exists():
                write_new(row_file, receipt)
            rows.append(row_summary)
            receipts.append(receipt)
            with (evidence / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical({k: receipt[k] for k in ("row", "count", "missing_tags", "overlay_copies", "wall_seconds", "cpu_seconds")}) + "\n")
            print(canonical({"row": key, "status": "validated", "count": expected_count, "missing_tags": missing}), flush=True)
        if seq != 23792 or sum(r["overlay_copies"] for r in receipts) != 23 or sum(len(r["excluded_old_kept"]) for r in receipts) != 178:
            raise ValueError("exact accepted/addition/exclusion totals failed")
        if {tuple(i) for r in receipts for i in r["overlay_identities"]} != set(additions):
            raise ValueError("overlay identity set mismatch")
        # Ensure all inputs, including PNG link counts, remained unchanged.
        for receipt in receipts:
            for path, info in receipt["files"].items():
                if metadata(Path(path)) != info["metadata"]:
                    raise ValueError(f"input metadata changed during build: {path}")
        for path, sha in pins.items():
            if digest(Path(path)) != sha:
                raise ValueError(f"historical artifact changed during build: {path}")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("built SQLite integrity failure")
        suggestions = Counter()
        for r in receipts:
            suggestions.update(r["suggestions"])
        categories = [r[0] for r in db.execute("SELECT DISTINCT category FROM vocabulary ORDER BY category")]
        # Resume state contains timings. Keep it only in the private journal;
        # the sealed index and release metadata are deterministic.
        db.execute("DROP TABLE build_rows")
        db.commit()
        db.execute("VACUUM")
    finally:
        db.close()
    missing = sum(r["missing_tags"] for r in receipts)
    manifest = {"schema": SCHEMA, "release": release_id, "count": seq,
        "index_sha256": digest(release / "index.sqlite"), "rows": rows, "categories": categories,
        "full_search_complete": missing == 0, "missing_tags": missing,
        "provenance_mode": aggregate["provenance_mode"], "accepted_pins": pins,
        "unchanged_accepted_row_pins": overlay["unchanged_accepted_row_pins"],
        "overlay_copies": 23, "existing_native_references": 23769, "excluded_old_kept": 178,
        "suggestions": [{"tag": tag, "category": "character", "count": count} for tag, count in sorted(suggestions.items(), key=lambda x: (-x[1], x[0]))[:30]],
        "tag_score_storage": "lossless pinned JSON references plus zlib little-endian float64 columns; no confidence cutoff",
        "row_provenance": [{k: r[k] for k in ("row", "report", "native_mapping", "source_version", "batch", "source_fingerprint")} for r in receipts]}
    write_new(release / "manifest.json", manifest)
    result = {"manifest": str(release / "manifest.json"), "manifest_sha256": digest(release / "manifest.json"),
        "index_sha256": manifest["index_sha256"], "index_size": (release / "index.sqlite").stat().st_size,
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu,
        "candidate_png_decodes": sum(r["candidate_png_decodes"] for r in receipts),
        "cache_bytes_read": sum(r["cache_bytes_read"] for r in receipts), "missing_tags": missing,
        "entries": seq, "pointer_written": False, "new_inference": False, "video_reads": 0}
    write_new(evidence / "build-result.json", result)
    print(canonical(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    build(args.evidence.absolute(), args.release_id, args.output_root.absolute(), args.resume)
