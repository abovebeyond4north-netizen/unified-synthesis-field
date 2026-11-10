
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
