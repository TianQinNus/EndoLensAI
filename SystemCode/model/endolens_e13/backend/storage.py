"""Single-process persistent records and immutable assets with a storage quota."""
import json
import os
import sqlite3
import threading
from pathlib import Path


class Store:
    def __init__(self, root, max_bytes=2*1024**3):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.owner = (self.root/"server.lock").open("a+b")
        self.owner.seek(0,2)
        if self.owner.tell() == 0:
            self.owner.write(b"0"); self.owner.flush()
        self.owner.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError("Storage already owned; run exactly one uvicorn worker, no --reload")
        self.db = sqlite3.connect(self.root/"records.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records(kind TEXT,id TEXT,task TEXT,idx INTEGER,data TEXT,PRIMARY KEY(kind,id))")
        self.db.execute("CREATE INDEX IF NOT EXISTS task_idx ON records(kind,task,idx)")
        self.db.commit()
        self.max_bytes = max_bytes
        self.used_bytes = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        for row in self.list("analysis"):
            if row["status"] in ("queued", "running"):
                row.update(status="failed", error="SERVER_RESTARTED:partial results retained; create a new task")
                self.put("analysis", row["task_id"], row)

    def path(self, relative):
        value = Path(relative)
        path = (self.root/value).resolve()
        if value.is_absolute() or ".." in value.parts or not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Unsafe asset path")
        return path

    def write(self, relative, data):
        with self.lock:
            if self.used_bytes+len(data) > self.max_bytes:
                raise ValueError("Storage quota reached; preserve results and choose a new storage directory")
            path = self.path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
            self.used_bytes += len(data)
        return path

    def put(self, kind, identity, data, task=None, idx=None):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?,?,?)",
                            (kind, identity, task, idx, json.dumps(data, allow_nan=False)))
            self.db.commit()

    def get(self, kind, identity):
        with self.lock:
            row = self.db.execute("SELECT data FROM records WHERE kind=? AND id=?", (kind, identity)).fetchone()
        if row is None:
            raise KeyError(identity)
        return json.loads(row[0])

    def list(self, kind, task=None):
        with self.lock:
            if task is None:
                rows = self.db.execute("SELECT data FROM records WHERE kind=? ORDER BY idx,id", (kind,)).fetchall()
            else:
                rows = self.db.execute("SELECT data FROM records WHERE kind=? AND task=? ORDER BY idx,id", (kind,task)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def update(self, identity, **updates):
        with self.lock:
            row = self.get("analysis", identity)
            row.update(updates)
            self.put("analysis", identity, row)
            return row

    def close(self):
        self.db.close()
        self.owner.close()
