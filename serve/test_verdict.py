"""The decision head's verdict route (/v1/verdict) and its loader: the JEV readout at the API.

    python -m unittest serve.test_verdict -v
"""
import json
import math
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, load_decision_head, serve
from serve.test_server import ThinkingEngine

ROOT = Path(__file__).resolve().parent.parent


class EmbeddingEngine(ThinkingEngine):
    """A deterministic dim-8 vector per prompt, the ids recorded (same shape as test_embeddings')."""

    def __init__(self, tok):
        super().__init__(tok)
        self.embeds = []

    def embed(self, ids, timeout_s=1800.0):
        self.embeds.append(list(ids))
        return [float((i + len(ids)) % 11) - 5.0 for i in range(8)]


def head_dict(dim=8):
    return {"path": "test", "w": [0.5] * dim, "mu": [0.0] * dim, "sd": [1.0] * dim,
            "b": 0.0, "a": 2.0, "c": 0.0, "threshold": 0.7, "model": "mock", "dim": dim}


class VerdictMath(unittest.TestCase):
    def test_p_correct_matches_the_formula(self):
        tok = ByteTokenizer()
        svc = Service(EmbeddingEngine(tok), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.decision_head = head_dict()
        svc.embed_text = lambda content, normalize=None: ([0.2] * 8, 42)   # a fixed vector
        out = svc.verdict("2+2?", "4")
        z = sum(0.5 * (0.2 - 0.0) / 1.0 for _ in range(8))                 # w.(h-mu)/sd + b
        want = 1.0 / (1.0 + math.exp(-(2.0 * z + 0.0)))
        self.assertAlmostEqual(out["p_correct"], want, places=12)
        self.assertEqual((out["threshold"], out["accept"]), (0.7, out["p_correct"] >= 0.7))

    def test_no_head_is_a_clear_error(self):
        tok = ByteTokenizer()
        svc = Service(EmbeddingEngine(tok), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        with self.assertRaisesRegex(ValueError, "no decision head"):
            svc.verdict("2+2?", "4")


class VerdictLoader(unittest.TestCase):
    def test_loads_the_study_schema_and_takes_the_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "head.json"
            p.write_text(json.dumps({"weights": {"w": [1.0, -2.0], "mu": [0.1, 0.2], "sd": [1.0, 2.0],
                                                   "b": 0.5, "platt": [8.0, -2.0]},
                                     "test": {"threshold": 0.87}, "model": "m"}))
            h = load_decision_head(str(p))
            self.assertEqual((h["dim"], h["threshold"], h["a"], h["c"]), (2, 0.87, 8.0, -2.0))
            self.assertEqual(h["model"], "m")

    def test_mismatched_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "head.json"
            p.write_text(json.dumps({"weights": {"w": [1.0, 2.0], "mu": [0.0], "sd": [1.0, 1.0],
                                                  "b": 0.0, "platt": [1.0, 0.0]}}))
            with self.assertRaises(ValueError):
                load_decision_head(str(p))


class VerdictRoute(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = EmbeddingEngine(self.tok)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.decision_head = head_dict()
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read().decode() or "null")

    def test_verdict(self):
        code, out = self.request("/v1/verdict", {"task": "2+2?", "candidate": "4"})
        self.assertEqual(code, 200, out)
        self.assertTrue(0.0 < out["p_correct"] < 1.0)
        self.assertEqual(out["accept"], out["p_correct"] >= out["threshold"])
        self.assertGreater(out["tokens"], 0)

    def test_bad_input_is_400(self):
        code, _ = self.request("/v1/verdict", {"task": "2+2?"})
        self.assertEqual(code, 400)
        code, _ = self.request("/v1/verdict", {"task": "", "candidate": "4"})
        self.assertEqual(code, 400)

    def test_no_head_is_400(self):
        self.svc.decision_head = None
        code, out = self.request("/v1/verdict", {"task": "2+2?", "candidate": "4"})
        self.assertEqual(code, 400)
        self.assertIn("no decision head", out["error"]["message"])


if __name__ == "__main__":
    unittest.main()
