# Research security / verification model

The SQLite hash chain is locally tamper-evident against accidental edits, not a cryptographically signed immutable record. Privileged database modification can reconstruct the entire chain. Holdout protection is implemented as a software contract in a shared process, not as a hardened isolation boundary. Native Lean must execute for a proof to count as kernel-verified. Do not treat numerical agreement or a generated `.lean` file as proof certification.
