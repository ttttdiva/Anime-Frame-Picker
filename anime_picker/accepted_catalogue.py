"""Trusted, read-only accepted frames. No model, video, or curation imports.

The offline builder pins the lossless tag caches and transposes their scores
into compressed float64 columns. Search reads only those immutable columns;
it never opens a candidate image or falls back to inference.
"""
from __future__ import annotations

import array
import base64
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import hmac
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sqlite3
import stat
import struct
import sys
import threading
import unicodedata
import zlib


SCHEMA = 1
RELEASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}\Z")
ID_RE = re.compile(r"[0-9a-f]{64}\Z")


class CatalogueError(ValueError):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def stable_id(row_key: str, raw_id: str, pts: int, fingerprint: str) -> str:
    # Identity uses the raw strings, including Unicode normalization form.
    return hashlib.sha256(canonical([row_key, raw_id, pts, fingerprint]).encode("utf-8")).hexdigest()


def signature(path: Path) -> tuple:
    s = path.lstat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)


def safe_file(root: Path, relative: str, suffix: str | None = None, verified: set | None = None) -> Path:
    """Reject alternate path syntaxes and every symlink/reparse component.

    A bulk caller may pass one ``verified`` set across many files. Each directory
    prefix is then checked (lstat plus containment) once; every file is still
    lstat-checked individually. Per-request callers pass nothing and check all.
    """
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or "%" in relative or "\x00" in relative
            or PureWindowsPath(relative).drive or relative.startswith("/")):
        raise CatalogueError("unsafe catalogue path")
    parts = relative.split("/")
    if any(p in ("", ".", "..") or p.rstrip(" .") != p for p in parts):
        raise CatalogueError("unsafe catalogue path")
    if suffix and PurePosixPath(relative).suffix.lower() != suffix:
        raise CatalogueError("unexpected catalogue file type")
    target = root
    for index, part in enumerate([None, *parts]):
        if part is not None:
            target = target / part
        is_directory = index < len(parts)
        prefix = tuple(parts[:index])
        if verified is not None and is_directory and prefix in verified:
            continue
        s = target.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & 0x400:
            raise CatalogueError("catalogue links/reparse points are forbidden")
        if is_directory and not stat.S_ISDIR(s.st_mode):
            raise CatalogueError("catalogue parent is not a directory")
        if verified is not None and is_directory:
            if not target.resolve().is_relative_to(root.resolve()):
                raise CatalogueError("catalogue file is not a regular contained file")
            verified.add(prefix)
    # With no link/reparse ancestor, a regular file is contained by construction.
    if not stat.S_ISREG(s.st_mode) or (verified is None and not target.resolve().is_relative_to(root.resolve())):
        raise CatalogueError("catalogue file is not a regular contained file")
    return target


def png_dimensions(payload: bytes) -> tuple[int, int]:
    if payload[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" or len(payload) < 33:
        raise CatalogueError("invalid native PNG")
    return struct.unpack(">II", payload[16:24])


def pack_scores(values: array.array) -> bytes:
    if sys.byteorder != "little":
        values = array.array("d", values)
        values.byteswap()
    return zlib.compress(values.tobytes(), 6)


def unpack_scores(payload: bytes, count: int) -> array.array:
    dec = zlib.decompressobj()
    data = dec.decompress(payload, count * 8 + 1)
    if len(data) != count * 8 or not dec.eof or dec.unused_data or dec.unconsumed_tail:
        raise CatalogueError("invalid score column")
    values = array.array("d")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def create_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        PRAGMA page_size=4096;
        CREATE TABLE frames(seq INTEGER PRIMARY KEY, stable_id TEXT UNIQUE NOT NULL,
          row_key TEXT NOT NULL, row_order INTEGER NOT NULL, pts INTEGER NOT NULL,
          raw_id TEXT NOT NULL, image TEXT NOT NULL, sha256 TEXT NOT NULL,
          width INTEGER NOT NULL, height INTEGER NOT NULL, metadata TEXT NOT NULL,
          UNIQUE(row_key,raw_id,pts));
        CREATE TABLE vocabulary(tag_id INTEGER PRIMARY KEY, category TEXT NOT NULL,
          name TEXT NOT NULL, search_key TEXT NOT NULL, UNIQUE(category,name));
        CREATE TABLE scores(tag_id INTEGER NOT NULL, row_key TEXT NOT NULL,
          first_seq INTEGER NOT NULL, count INTEGER NOT NULL, data BLOB NOT NULL,
          PRIMARY KEY(tag_id,row_key)) WITHOUT ROWID;
        CREATE INDEX frame_row ON frames(row_key,seq);
        CREATE INDEX vocabulary_search ON vocabulary(category,search_key);
    """)


def validate_manifest(m: dict) -> None:
    """Validate public summary/filter fields before caching any release."""
    if (type(m["count"]) is not int or m["count"] < 1 or type(m["missing_tags"]) is not int
            or not 0 <= m["missing_tags"] <= m["count"]
            or type(m["full_search_complete"]) is not bool
            or m["full_search_complete"] != (m["missing_tags"] == 0)
            or not isinstance(m["provenance_mode"], str)
            or not isinstance(m["rows"], list) or not isinstance(m["categories"], list)
            or not isinstance(m["suggestions"], list)):
        raise CatalogueError("invalid catalogue manifest schema")
    keys = set()
    for row in m["rows"]:
        if (not isinstance(row["key"], str) or not 1 <= len(row["key"]) <= 80
                or row["key"] in keys or not isinstance(row["name"], str)
                or type(row["order"]) is not int or type(row["count"]) is not int
                or row["count"] < 0):
            raise CatalogueError("invalid catalogue row schema")
        keys.add(row["key"])
    if sum(row["count"] for row in m["rows"]) != m["count"]:
        raise CatalogueError("catalogue row totals mismatch")
    if any(not isinstance(c, str) or not 1 <= len(c) <= 80 for c in m["categories"]):
        raise CatalogueError("invalid catalogue category schema")
    for suggestion in m["suggestions"]:
        if (not isinstance(suggestion["tag"], str) or suggestion["category"] not in m["categories"]
                or type(suggestion["count"]) is not int or not 0 <= suggestion["count"] <= m["count"]):
            raise CatalogueError("invalid catalogue suggestion schema")


class AcceptedCatalogue:
    """One cached release; connections are request-local and SQLite read-only."""
    def __init__(self, output_root: Path, *, trusted_release=None, manifest_sha256=None):
        self.root = Path(output_root).absolute()
        self.lock = threading.RLock()
        self.pointer_bytes = None
        self.manifest = None
        self.manifest_sha = None
        self.pinned = []
        self.queries = OrderedDict()
        # Operator/test-only preview pin. HTTP callers cannot select a release.
        self.preview = None if trusted_release is None else canonical({
            "schema": SCHEMA, "release": trusted_release, "manifest_sha256": manifest_sha256}).encode()

    @contextmanager
    def _connect(self, index=None):
        db = sqlite3.connect((index or self.index).as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA trusted_schema=OFF")
            yield db
        finally:
            db.close()

    def _load(self) -> bool:
        pointer = self.root / "_accepted/current.json"
        if self.preview is None and not pointer.exists():
            # A dangling link is corruption, not an absent catalogue.
            if pointer.is_symlink():
                raise CatalogueError("invalid catalogue pointer")
            self.manifest = None
            self.pointer_bytes = None
            return False
        try:
            raw = self.preview
            if raw is None:
                pointer = safe_file(self.root, "_accepted/current.json", ".json")
                raw = pointer.read_bytes()
            if raw == self.pointer_bytes and self.manifest is not None:
                for path, sig in self.pinned:
                    safe_file(self.root, path.relative_to(self.root).as_posix())
                    if signature(path) != sig:
                        raise CatalogueError("catalogue integrity drift")
                return True
            p = json.loads(raw)
            release = p["release"]
            if p["schema"] != SCHEMA or not isinstance(release, str) or not RELEASE_RE.fullmatch(release):
                raise CatalogueError("invalid release pointer")
            manifest_path = safe_file(self.root, f"_accepted/{release}/manifest.json", ".json")
            manifest_raw = manifest_path.read_bytes()
            manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
            if not hmac.compare_digest(manifest_sha, p["manifest_sha256"]):
                raise CatalogueError("catalogue manifest hash mismatch")
            m = json.loads(manifest_raw)
            if m["schema"] != SCHEMA or m["release"] != release or m["count"] < 1:
                raise CatalogueError("invalid catalogue manifest")
            validate_manifest(m)
            index = safe_file(self.root, f"_accepted/{release}/index.sqlite", ".sqlite")
            if digest(index) != m["index_sha256"]:
                raise CatalogueError("catalogue index hash mismatch")
            with self._connect(index) as db:
                # The builder ran integrity_check before sealing this exact SHA.
                # Rewalking every score BLOB here makes cold UI startup very slow.
                if db.execute("SELECT count(*) FROM frames").fetchone()[0] != m["count"]:
                    raise CatalogueError("catalogue count mismatch")
                # Validate all image references once, without reading PNG payloads.
                # Each shared directory is checked once; every file is still lstat-checked.
                verified = set()
                for f in db.execute("SELECT image,stable_id FROM frames"):
                    if not ID_RE.fullmatch(f["stable_id"]):
                        raise CatalogueError("invalid frame ID")
                    safe_file(self.root, f["image"], ".png", verified)
            self.manifest = m
            self.index = index
            self.manifest_sha = manifest_sha
            self.pointer_bytes = raw
            self.pinned = [(f, signature(f)) for f in (manifest_path, self.index)]
            self.queries.clear()
            return True
        except CatalogueError:
            raise
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            raise CatalogueError("accepted catalogue is corrupt or unavailable") from exc

    def prewarm(self) -> bool:
        """Run the full pin verification ahead of the first request.

        A failure is not cached: the next request repeats the load and reports
        the explicit error.
        """
        try:
            with self.lock:
                return self._load()
        except CatalogueError:
            return False

    def summary(self) -> dict:
        with self.lock:
            if not self._load():
                return {"available": False, "count": 0, "rows": [], "categories": [],
                        "full_search_complete": False, "message": "No accepted catalogue installed"}
            m = self.manifest
            return {k: m[k] for k in ("release", "count", "rows", "categories", "full_search_complete",
                                      "missing_tags", "provenance_mode", "suggestions")} | {
                "available": True, "min_score_default": 0.1,
                "tag_evidence": "Cached accepted frame tags; no identity inference"}

    def _matches(self, db, q: str, row: str, category: str, score: float):
        key = (q, row, category, score)
        if key in self.queries:
            self.queries.move_to_end(key)
            return self.queries[key]
        # LIKE metacharacters are literal user text; only our outer '%' match.
        pattern = "%" + q.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"
        sql = "SELECT tag_id,category,name FROM vocabulary WHERE search_key LIKE ? ESCAPE '!'"
        args = [pattern]
        if category:
            sql += " AND category=?"
            args.append(category)
        matches = set()
        for tag in db.execute(sql, args):
            column_sql = "SELECT first_seq,count,data FROM scores WHERE tag_id=?"
            column_args = [tag["tag_id"]]
            if row:
                column_sql += " AND row_key=?"
                column_args.append(row)
            for block in db.execute(column_sql, column_args):
                if (type(block["count"]) is not int or not 1 <= block["count"] <= self.manifest["count"]
                        or type(block["first_seq"]) is not int or block["first_seq"] < 0
                        or block["first_seq"] + block["count"] > self.manifest["count"]):
                    raise CatalogueError("invalid score column bounds")
                values = unpack_scores(block["data"], block["count"])
                matches.update(block["first_seq"] + i for i, value in enumerate(values)
                               if math.isfinite(value) and value >= score)
        result = tuple(sorted(matches))
        self.queries[key] = result
        if len(self.queries) > 16:
            self.queries.popitem(last=False)
        return result

    def frames(self, *, q="", row="", tag_category="", min_score=0.1,
               cursor="", limit=40) -> dict:
        if (not isinstance(q, str) or len(q) > 256 or "\x00" in q
                or not isinstance(row, str) or len(row) > 80
                or not isinstance(tag_category, str) or len(tag_category) > 80
                or not isinstance(cursor, str) or len(cursor) > 2048
                or type(limit) is not int or not 1 <= limit <= 100
                or isinstance(min_score, bool) or not isinstance(min_score, (int, float))
                or not math.isfinite(min_score) or not 0 <= min_score <= 1):
            raise CatalogueError("invalid catalogue search parameters", 422)
        q = normalized(q.strip())
        with self.lock:
            if not self._load():
                if cursor or row or tag_category:
                    raise CatalogueError("catalogue/filter not found", 404)
                return {"items": [], "total": 0, "next_cursor": None, "offset": 0, "release": None}
            m = self.manifest
            if row and row not in {r["key"] for r in m["rows"]}:
                raise CatalogueError("unknown catalogue row", 404)
            if tag_category and tag_category not in m["categories"]:
                raise CatalogueError("unknown tag category", 422)
            identity = hashlib.sha256(canonical([self.manifest_sha, q, row, tag_category, float(min_score), limit]).encode()).hexdigest()
            offset = 0
            if cursor:
                try:
                    decoded = base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True)
                    c = json.loads(decoded)
                    if not isinstance(c, dict) or set(c) != {"query", "offset", "signature"}:
                        raise ValueError()
                    body = canonical([c["query"], c["offset"]])
                    expected = hmac.new(self.manifest_sha.encode(), body.encode(), "sha256").hexdigest()
                    if c["query"] != identity or not hmac.compare_digest(expected, c["signature"]):
                        raise ValueError()
                    offset = c["offset"]
                    if type(offset) is not int or offset < 0:
                        raise ValueError()
                except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                    raise CatalogueError("malformed, stale or mismatched catalogue cursor", 400) from exc
            with self._connect() as db:
                if q:
                    seqs = self._matches(db, q, row, tag_category, min_score)
                    total = len(seqs)
                    chosen = seqs[offset:offset + limit]
                    # Bounded page; no user strings are interpolated into SQL.
                    rows = [db.execute("SELECT * FROM frames WHERE seq=?", (seq,)).fetchone() for seq in chosen]
                else:
                    clause, params = (" WHERE row_key=?", [row]) if row else ("", [])
                    total = db.execute("SELECT count(*) FROM frames" + clause, params).fetchone()[0]
                    rows = db.execute("SELECT * FROM frames" + clause + " ORDER BY seq LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
                if offset > total or (cursor and offset == total):
                    raise CatalogueError("catalogue cursor offset out of range", 400)
                items = []
                for f in rows:
                    metadata = json.loads(f["metadata"])
                    items.append({k: f[k] for k in ("stable_id", "row_key", "pts", "raw_id", "width", "height")} | {
                        "time": metadata["time"], "source_name": metadata["source_name"],
                        "image_url": f'/api/accepted-catalogue/frames/{f["stable_id"]}/image',
                        "native_origin": metadata["native_origin"], "tags_available": metadata["tag_evidence"]["status"] == "complete",
                        "cut_quality": metadata["accepted_record"].get("cut_quality")})
            next_cursor = None
            if offset + len(items) < total:
                body = canonical([identity, offset + len(items)])
                signature_ = hmac.new(self.manifest_sha.encode(), body.encode(), "sha256").hexdigest()
                next_cursor = base64.urlsafe_b64encode(canonical({"query": identity, "offset": offset + len(items), "signature": signature_}).encode()).decode()
            return {"items": items, "total": total, "offset": offset, "next_cursor": next_cursor,
                    "release": m["release"], "full_search_complete": m["full_search_complete"], "min_score": min_score}

    def image(self, frame_id: str) -> bytes:
        with self.lock:
            if not ID_RE.fullmatch(frame_id) or not self._load():
                raise CatalogueError("accepted frame not found", 404)
            with self._connect() as db:
                f = db.execute("SELECT image,sha256,width,height FROM frames WHERE stable_id=?", (frame_id,)).fetchone()
            if f is None:
                raise CatalogueError("accepted frame not found", 404)
            try:
                path = safe_file(self.root, f["image"], ".png")
                payload = path.read_bytes()
                if (hashlib.sha256(payload).hexdigest() != f["sha256"]
                        or png_dimensions(payload) != (f["width"], f["height"])):
                    raise CatalogueError("accepted native image integrity mismatch")
                return payload
            except OSError as exc:
                raise CatalogueError("accepted native image unavailable") from exc
