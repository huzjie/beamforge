"""A tiny OpenAI-compatible client (stdlib urllib)."""
import json
import urllib.request


class BeamClient:
    def __init__(self, base_url="http://127.0.0.1:8899"):
        self.base_url = base_url.rstrip("/")

    def health(self):
        return json.loads(urllib.request.urlopen(f"{self.base_url}/health", timeout=10).read())

    def chat(self, content):
        body = json.dumps({"messages": [{"role": "user", "content": content}]}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return data["choices"][0]["message"]["content"]
