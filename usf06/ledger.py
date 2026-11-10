"""Append-only SQLite event journal with hash-chain verification.

No external anchoring/signature: a privileged attacker can rewrite the entire DB.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .core import canonical_json, sha256

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS ledger_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    subject_id TEXT,
    parent_id TEXT,
    event_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL UNIQUE,
    created_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_events ON ledger_events(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_subject ON ledger_events(subject_id);
CREATE TRIGGER IF NOT EXISTS prevent_ledger_update
  BEFORE UPDATE ON ledger_events BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS prevent_ledger_delete
  BEFORE DELETE ON ledger_events BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
-- Evaluations are separate append-only events; no post-hoc mutation of candidate records.
CREATE VIEW IF NOT EXISTS discovery_ledger AS
 SELECT sequence, run_id, subject_id AS candidate_id, parent_id,
        json_extract(event_json, '$.generation') AS generation,
        json_extract(event_json, '$.expression') AS ast_expression,
        json_extract(event_json, '$.mdl_bits') AS mdl_cost_bits,
        json_extract(event_json, '$.train_exact') AS train_exact_fraction,
        json_extract(event_json, '$.falsification_survived') AS falsification_survived,
        json_extract(event_json, '$.logos_status') AS logos_status,
        json_extract(event_json, '$.axiom_audit_pass') AS axiom_audit_pass,
        record_hash
 FROM ledger_events WHERE event_type = 'candidate';
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA synchronous = FULL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def append(self, run_id: str, event_type: str, payload: dict,
               subject_id: str | None = None, parent_id: str | None = None):
        if not run_id or not event_type:
            raise ValueError("run_id and event_type are required")
        event_json = canonical_json(payload)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if event_type == "candidate":
                if not subject_id:
                    raise ValueError("Candidate must have a unique subject ID")
                duplicate = self.db.execute(
                    "SELECT 1 FROM ledger_events WHERE run_id=? AND event_type='candidate' AND subject_id=? LIMIT 1",
                    (run_id, subject_id)).fetchone()
                if duplicate:
                    raise ValueError("Duplicate candidate ID within run")
                if parent_id:
                    ancestor = self.db.execute(
                        "SELECT 1 FROM ledger_events WHERE run_id=? AND event_type='candidate' AND subject_id=? LIMIT 1",
                        (run_id, parent_id)).fetchone()
                    if not ancestor:
                        raise ValueError("Missing parent candidate within run")
            last = self.db.execute("SELECT record_hash FROM ledger_events ORDER BY sequence DESC LIMIT 1").fetchone()
            prev = last[0] if last else "GENESIS"
            now = datetime.now(timezone.utc).isoformat()
            record = {"run_id": run_id, "event_type": event_type, "subject_id": subject_id,
                      "parent_id": parent_id, "event_json": event_json, "previous_hash": prev,
                      "created_utc": now}
            digest = sha256(record)
            self.db.execute("""INSERT INTO ledger_events
                (run_id,event_type,subject_id,parent_id,event_json,previous_hash,record_hash,created_utc)
                VALUES (?,?,?,?,?,?,?,?)""",
                (run_id,event_type,subject_id,parent_id,event_json,prev,digest,now))
            self.db.commit()
            return digest
        except BaseException:
            self.db.rollback()
            raise

    def verify_integrity(self):
        prev = "GENESIS"
        count = 0
        for seq, run_id, kind, subject, parent, event_json, stored_prev, digest, created_utc in self.db.execute(
            """SELECT sequence,run_id,event_type,subject_id,parent_id,event_json,previous_hash,record_hash,created_utc
               FROM ledger_events ORDER BY sequence"""):
            if stored_prev != prev or canonical_json(json.loads(event_json)) != event_json:
                return False, seq
            recomputed = sha256({"run_id": run_id, "event_type": kind, "subject_id": subject,
                                 "parent_id": parent, "event_json": event_json, "previous_hash": stored_prev,
                                 "created_utc": created_utc})
            if recomputed != digest:
                return False, seq
            prev = digest
            count += 1
        return True, count

    def events(self, run_id: str | None = None):
        if run_id is None:
            rows = self.db.execute("SELECT event_type,subject_id,event_json FROM ledger_events ORDER BY sequence")
        else:
            rows = self.db.execute("SELECT event_type,subject_id,event_json FROM ledger_events WHERE run_id=? ORDER BY sequence", (run_id,))
        return [{"event_type": a, "subject_id": b, "payload": json.loads(c)} for a, b, c in rows]
