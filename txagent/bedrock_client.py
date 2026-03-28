import os
import json
import time
import itertools
import threading
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None
    BotoConfig = None


def extract_bedrock_text(resp: Dict[str, Any]) -> str:
    """Extract human-readable text returned by the Bedrock runtime."""
    if isinstance(resp, dict):
        output = resp.get("output") or {}
        if isinstance(output, dict):
            message = output.get("message") or {}
            for block in message.get("content", []):
                if "text" in block:
                    return block["text"]
        if "outputText" in resp:
            return resp["outputText"]
    return json.dumps(resp)[:1000]


class BedrockLLM:
    """Minimal Bedrock runtime wrapper with a rotating client pool."""

    def __init__(
        self,
        model_id: str,
        region: Optional[str] = None,
        pool_size: int = 3,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if boto3 is None:
            raise ImportError("boto3 is required for the Bedrock backend.")
        region_name = region or os.environ.get("AWS_REGION") or "us-west-2"
        cfg = BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=180,
            max_pool_connections=25,
        )
        client_kwargs = client_kwargs or {}
        self.model_id = model_id
        self.clients = [
            boto3.client("bedrock-runtime", region_name=region_name, config=cfg, **client_kwargs)
            for _ in range(max(1, pool_size))
        ]
        self._cycle = itertools.cycle(self.clients)
        self._lock = threading.Lock()

    def _client(self):
        with self._lock:
            return next(self._cycle)

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        params = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
                "topP": 0.9,
            },
        }
        for attempt in range(3):
            try:
                resp = self._client().converse(**params)
                return extract_bedrock_text(resp)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return "(bedrock request failed)"
