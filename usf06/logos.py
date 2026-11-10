"""Native Lean 4 + Mathlib proof gate; FAILS CLOSED if tooling/audit unavailable."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from .core import Expr

_ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


@dataclass(frozen=True)
class FormalResult:
    status: str
    kernel_verified: bool
    axiom_audit_pass: bool
    lean_source: str
    diagnostics: str
    elapsed_s: float
    lean_version: str
    axioms: tuple[str, ...] = ()


def parse_axiom_output(stdout: str, theorem: str) -> tuple[str, ...] | None:
    # Only accept a result naming THIS theorem. Ignore unexpected or absent output.
    match = re.search(r"['\u2018]" + re.escape(theorem) +
                      r"['\u2019] depends on axioms:\s*\[([^\]]*)\]", stdout)
    if match:
        return tuple(a.strip() for a in match.group(1).split(",") if a.strip())
    if re.search(r"['\u2018]" + re.escape(theorem) +
                 r"['\u2019] does not depend on any axioms", stdout):
        return ()
    return None


def build_lean(candidate: Expr, target: Expr, theorem: str) -> str:
    if not re.fullmatch(r"usf_[a-z0-9_]+", theorem):
        raise ValueError("Invalid generated theorem identifier")
    for ast in (candidate, target):
        if ast.size() > 120 or ast.depth() > 30:
            raise ValueError("Lean AST budget exceeded")
    return ("import Mathlib.Tactic.Ring\n"
            "set_option autoImplicit false\n\n"
            f"theorem {theorem} (x : ℤ) : {candidate.to_lean()} = {target.to_lean()} := by\n"
            "  ring\n\n"
            f"#print axioms {theorem}\n")


class LogosGate:
    def __init__(self, mathlib_project: str | None = None, timeout_s: float = 20):
        self.project = Path(mathlib_project).resolve() if mathlib_project else None
        self.timeout_s = timeout_s
        self.lake = shutil.which("lake")
        self.version = "UNAVAILABLE"
        if self.lake and self.project and (self.project / "lakefile.toml").exists() or (
            self.lake and self.project and (self.project / "lakefile.lean").exists()
        ):
            try:
                p = subprocess.run([self.lake, "env", "lean", "--version"],
                                   cwd=self.project, capture_output=True, text=True, timeout=10)
                if p.returncode == 0:
                    self.version = p.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass

    def verify(self, candidate: Expr, target: Expr, theorem: str) -> FormalResult:
        script = build_lean(candidate, target, theorem)
        t0 = time.perf_counter()
        if self.version == "UNAVAILABLE":
            return FormalResult("LEAN_UNAVAILABLE", False, False, script,
                                "Requires a Mathlib Lake project and native `lake env lean`.",
                                time.perf_counter() - t0, self.version)
        with tempfile.TemporaryDirectory(prefix="usf_lean_") as tmp:
            path = Path(tmp) / f"{theorem}.lean"
            path.write_text(script, encoding="utf-8")
            try:
                p = subprocess.run([self.lake, "env", "lean", str(path)],
                                   cwd=self.project, capture_output=True, text=True,
                                   timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return FormalResult("LEAN_TIMEOUT", False, False, script, "Timeout",
                                    time.perf_counter() - t0, self.version)
            except OSError as exc:
                return FormalResult("LEAN_LAUNCH_FAILED", False, False, script, str(exc),
                                    time.perf_counter() - t0, self.version)
        output = p.stdout + "\n" + p.stderr
        if p.returncode != 0 or re.search(r"\b(?:error:|sorryAx|admit|sorry)\b", output, re.I):
            return FormalResult("LEAN_FAILED", False, False, script, output[-3000:],
                                time.perf_counter() - t0, self.version)
        axioms = parse_axiom_output(output, theorem)
        if axioms is None:
            return FormalResult("AUDIT_MISSING", False, False, script, output[-3000:],
                                time.perf_counter() - t0, self.version)
        if not set(axioms).issubset(_ALLOWED_AXIOMS):
            return FormalResult("AXIOMS_REJECTED", False, False, script, output[-3000:],
                                time.perf_counter() - t0, self.version, axioms)
        return FormalResult("KERNEL_VERIFIED", True, True, script,
                            output[-3000:], time.perf_counter()-t0, self.version, axioms)
