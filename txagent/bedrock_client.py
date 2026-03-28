import os
import json
import time
import itertools
import threading
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:  # pragma: no cover - optional dependency
    boto3 = None
    BotoConfig = None


# ── format converters ─────────────────────────────────────────────────────────

def _tools_to_bedrock(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert TxAgent tool list → Bedrock toolConfig dict."""
    specs = []
    for t in tools:
        name = t.get("name") or t.get("tool_name", "")
        desc = t.get("description", "")
        params = t.get("parameters") or t.get("arguments") or {}
        # Ensure params is a valid JSON-schema object block
        if not isinstance(params, dict) or params.get("type") != "object":
            params = {"type": "object", "properties": params if isinstance(params, dict) else {}}
        specs.append({
            "toolSpec": {
                "name": name,
                "description": desc,
                "inputSchema": {"json": params},
            }
        })
    return {"tools": specs}


def _messages_to_bedrock(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert TxAgent conversation messages → Bedrock Converse message list.

    TxAgent roles:
      system   → stripped out (passed separately as systemPrompt)
      user     → user message with text block
      assistant→ assistant message; if tool_calls present, includes toolUse blocks
      tool     → converted to user message with toolResult blocks
    """
    bedrock_msgs = []
    pending_tool_results: List[Dict] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            continue  # handled separately

        if role == "tool":
            # Accumulate tool results; they'll be flushed as a single user message
            raw = msg.get("content", "{}")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                parsed = {"content": str(raw)}
            call_id = parsed.get("call_id", "tool_0")
            result_text = str(parsed.get("content", ""))
            pending_tool_results.append({
                "toolResult": {
                    "toolUseId": call_id,
                    "content": [{"text": result_text}],
                }
            })
            continue

        # Flush pending tool results before any non-tool message
        if pending_tool_results:
            bedrock_msgs.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

        if role == "assistant":
            content_blocks = []
            text = (msg.get("content") or "").strip()
            if text:
                content_blocks.append({"text": text})

            raw_calls = msg.get("tool_calls")
            if raw_calls:
                try:
                    calls = json.loads(raw_calls) if isinstance(raw_calls, str) else raw_calls
                    if isinstance(calls, dict):
                        calls = [calls]
                    for call in (calls or []):
                        call_id = call.get("call_id") or f"call_{len(content_blocks)}"
                        content_blocks.append({
                            "toolUse": {
                                "toolUseId": call_id,
                                "name": call.get("name", ""),
                                "input": call.get("arguments") or {},
                            }
                        })
                except (json.JSONDecodeError, TypeError):
                    pass

            if content_blocks:
                bedrock_msgs.append({"role": "assistant", "content": content_blocks})

        else:  # user
            text = (msg.get("content") or "")
            if not isinstance(text, str):
                text = str(text)
            bedrock_msgs.append({"role": "user", "content": [{"text": text}]})

    # Flush any remaining tool results
    if pending_tool_results:
        bedrock_msgs.append({"role": "user", "content": pending_tool_results})

    return bedrock_msgs


def _parse_bedrock_response(resp: Dict[str, Any]) -> str:
    """
    Parse a Bedrock Converse response into a TxAgent-compatible string.

    If the response contains toolUse blocks, the result is formatted as:
        <reasoning text>
        [TOOL_CALLS]
        [{"name": "...", "arguments": {...}, "call_id": "..."}]

    Otherwise returns the plain text response.
    """
    message = (resp.get("output") or {}).get("message") or {}
    content_blocks = message.get("content") or []

    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        if "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append({
                "name": tu.get("name", ""),
                "arguments": tu.get("input") or {},
                "call_id": tu.get("toolUseId", ""),
            })

    reasoning = "\n".join(text_parts).strip()

    if tool_calls:
        return f"{reasoning}\n[TOOL_CALLS]\n{json.dumps(tool_calls)}"
    return reasoning or json.dumps(resp)[:1000]


# ── client ────────────────────────────────────────────────────────────────────

class BedrockLLM:
    """Bedrock runtime wrapper with native tool-use support."""

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

    def _call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(3):
            try:
                return self._client().converse(**params)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Bedrock converse failed after retries.")

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        """Plain text chat — no tool calling."""
        params = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
                "topP": 0.9,
            },
        }
        resp = self._call(params)
        return _parse_bedrock_response(resp)

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """
        Structured chat using Bedrock's native tool-use API.
        Accepts TxAgent-format messages and tools; returns a TxAgent-format string
        (plain text, or 'reasoning\\n[TOOL_CALLS]\\n[{...}]' when tools are called).
        """
        bedrock_messages = _messages_to_bedrock(messages)

        params: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
                "topP": 0.9,
            },
        }
        if system_prompt:
            params["system"] = [{"text": system_prompt}]
        if tools:
            params["toolConfig"] = _tools_to_bedrock(tools)

        resp = self._call(params)
        return _parse_bedrock_response(resp)
