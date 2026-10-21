"""The EMB request must not kill the engine: the readback happens after the window's cleanup.

The EMB request (`GEN 1 emb=<file> ...`) reads the last prompt token's head
input and ends the request without emitting a token.  Its first
implementation broke out of the decode loop *before* the window's cleanup
(the adaptive tier's thread join and the draft's commit), so the adaptive
thread was still running when the request ended.  The engine then died with
"terminate called without an active exception" on the request *after* the
first EMB one - a crash that only showed up in a long run.

This checks the engine survives a run of EMB requests interleaved with
normal ones.  It needs a built engine (`build-hip/strata`) and a model;
without them it skips rather than fails.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build-hip" / "strata"


def _have_engine_and_model() -> bool:
    if not ENGINE.is_file():
        return False
    model = os.environ.get("STRATA_TEST_MODEL", "")
    return bool(model) and Path(model).is_file()


@unittest.skipUnless(_have_engine_and_model(), "no built engine and model (STRATA_TEST_MODEL)")
class TestEmbSurvives(unittest.TestCase):
    """A run of EMB requests, then a normal one: the engine must still answer."""

    def test_emb_then_normal_request(self):
        model = os.environ["STRATA_TEST_MODEL"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "emb.f32"
            proc = subprocess.Popen(
                [str(ENGINE), "--serve", "--native", model, "--max-context", "2048",
                 "--port", "18997"],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE, text=True,
            )
            try:
                self._wait_ready(proc)
                ids = "9707,1917,3838"   # "What is 2+2?"
                # five EMB requests in a row: the crash used to come after the first
                for _ in range(5):
                    proc.stdin.write(f"GEN 1 emb={out} {ids}\n")
                    proc.stdin.flush()
                    line = self._read_line(proc)
                    self.assertTrue(line.startswith("EMB "), f"the EMB answer was {line!r}")
                    self.assertTrue(out.is_file(), "the EMB vector was not written")
                # and a normal request still works: the engine is alive
                proc.stdin.write(f"GEN 8 {ids}\n")
                proc.stdin.flush()
                tokens = []
                while True:
                    line = self._read_line(proc)
                    if line.startswith("T "):
                        tokens.append(line)
                    elif line.startswith("DONE"):
                        break
                self.assertTrue(tokens, "the engine answered nothing after the EMB requests")
            finally:
                try:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                time.sleep(1)
                proc.kill()

    def _read_line(self, proc):
        line = proc.stdout.readline()
        self.assertTrue(line, "the engine closed its output (it died)")
        return line.strip()

    def _wait_ready(self, proc):
        deadline = time.time() + 300
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                raise AssertionError("the engine died on startup")
            if "ready:" in line:
                return
        raise AssertionError("the engine did not come up in 5 minutes")


if __name__ == "__main__":
    unittest.main()
