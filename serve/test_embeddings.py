"""The text-embedding endpoints (/embedding, /v1/embeddings) and /apply-template: the decision-head readout's
vector, through the HTTP API (no GPU or model pack).

    python -m unittest serve.test_embeddings -v
"""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, serve
from serve.test_server import ThinkingEngine

ROOT = Path(__file__).resolve().parent.parent


class EmbeddingEngine(ThinkingEngine):
    """The engine's emb= answer: one deterministic vector per prompt, the ids recorded."""

    def __init__(self, tok):
        super().__init__(tok)
        self.embeds = []

    def embed(self, ids, timeout_s=1800.0):
        self.embeds.append(list(ids))
        return [float((i + len(ids)) % 11) - 5.0 for i in range(8)]


class Embeddings(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = EmbeddingEngine(self.tok)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
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

    def test_apply_template_renders_the_model_template(self):
        code, out = self.request("/apply-template", {"messages": [{"role": "user", "content": "2+2?"}],
                                                     "add_generation_prompt": True})
        self.assertEqual(code, 200)
        self.assertIn("2+2?", out["prompt"])

    def test_apply_template_rejects_no_messages(self):
        code, _ = self.request("/apply-template", {"messages": []})
        self.assertEqual(code, 400)

    def test_llamacpp_shape_is_raw(self):
        code, out = self.request("/embedding", {"content": "hello", "embd_normalize": -1})
        self.assertEqual(code, 200)
        self.assertIsInstance(out, list)
        vec = out[0]["embedding"][0]
        self.assertEqual(len(vec), 8)
        self.assertNotAlmostEqual(sum(v * v for v in vec) ** 0.5, 1.0, places=3)

    def test_openai_shape_is_unit_norm(self):
        code, out = self.request("/v1/embeddings", {"model": "m", "input": "hello"})
        self.assertEqual(code, 200)
        vec = out["data"][0]["embedding"]
        self.assertAlmostEqual(sum(v * v for v in vec) ** 0.5, 1.0, places=6)
        self.assertEqual(out["usage"]["prompt_tokens"], len(self.engine.embeds[0]))

    def test_l2_on_the_llamacpp_shape_too(self):
        code, out = self.request("/embedding", {"content": "hello", "embd_normalize": 2})
        self.assertEqual(code, 200)
        vec = out[0]["embedding"][0]
        self.assertAlmostEqual(sum(v * v for v in vec) ** 0.5, 1.0, places=6)

    def test_empty_content_is_400(self):
        code, _ = self.request("/embedding", {"content": "", "embd_normalize": -1})
        self.assertEqual(code, 400)

    def test_bad_normalize_is_400(self):
        code, _ = self.request("/embedding", {"content": "hi", "embd_normalize": 7})
        self.assertEqual(code, 400)

    def test_engine_without_embed_is_400(self):
        self.engine.embed = None
        code, out = self.request("/embedding", {"content": "hi", "embd_normalize": -1})
        self.assertEqual(code, 400)
        self.assertIn("cannot embed", out["error"]["message"])


if __name__ == "__main__":
    unittest.main()
