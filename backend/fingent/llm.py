"""
LLM provider — an OpenAI-compatible chat-completions client.

This is the platform's reasoning backend. It defaults to Groq (Llama) but is provider-agnostic:
point FINGENT_LLM_BASE_URL at any OpenAI-compatible endpoint (Groq, OpenAI, Azure OpenAI, a
self-hosted vLLM/Ollama gateway, etc.) and the runtime + compiler use it unchanged. This is what
lets a financial-services tenant run agents against their own approved model gateway.

The runtime uses native **tool calling** (`tools` / `tool_calls`) so the model selects tools and
fills their arguments from each tool's JSON schema — including MCP-discovered tools. When no key
is configured the platform falls back to the deterministic demo engine (clearly flagged), so it
still runs offline. `stream_chat` adds token-level Server-Sent-Events streaming for the chat UI.

Config (first provider with a key wins; FINGENT_LLM_* always overrides):
  Generic  : FINGENT_LLM_API_KEY  + FINGENT_LLM_BASE_URL + FINGENT_LLM_MODEL
  Groq     : GROQ_API_KEY         (base https://api.groq.com/openai/v1, model llama-3.3-70b-versatile)
  Gemini   : GEMINI_API_KEY       (Google's OpenAI-compatible endpoint, model gemini-2.0-flash)

Gemini is used via its OpenAI-compatibility layer
(https://generativelanguage.googleapis.com/v1beta/openai), so native tool-calling, streaming and
JSON response_format all work through the same code path — no Google SDK needed.
"""
from __future__ import annotations

import json
import os

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"


def _resolve_provider() -> tuple[str, str, str]:
    """Pick (api_key, base_url, model) from the environment. An explicit FINGENT_LLM_* generic
    config wins; otherwise the first provider whose key is set is used (Groq, then Gemini)."""
    base_override = os.getenv("FINGENT_LLM_BASE_URL")
    model_override = os.getenv("FINGENT_LLM_MODEL")

    if os.getenv("FINGENT_LLM_API_KEY"):
        return (os.getenv("FINGENT_LLM_API_KEY"),
                (base_override or DEFAULT_BASE_URL).rstrip("/"),
                model_override or DEFAULT_MODEL)
    if os.getenv("GROQ_API_KEY"):
        return (os.getenv("GROQ_API_KEY"),
                (base_override or os.getenv("GROQ_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
                model_override or os.getenv("GROQ_MODEL") or DEFAULT_MODEL)
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                (base_override or os.getenv("GEMINI_BASE_URL") or GEMINI_BASE_URL).rstrip("/"),
                model_override or os.getenv("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL)
    return "", (base_override or DEFAULT_BASE_URL).rstrip("/"), (model_override or DEFAULT_MODEL)


def _retry_after(resp, attempt: int) -> float:
    """Seconds to wait before a retry: honour the server's Retry-After header, else exponential
    backoff (1.5, 3, 6 s) capped at 8s."""
    ra = (resp.headers.get("Retry-After") or "").strip()
    try:
        if ra:
            return min(float(ra), 8.0)
    except ValueError:
        pass
    return min(1.5 * (2 ** attempt), 8.0)


class LlmProvider:
    def __init__(self) -> None:
        self.api_key, self.base_url, self.model = _resolve_provider()
        self.timeout = float(os.getenv("FINGENT_LLM_TIMEOUT", "40"))
        self.max_tokens = int(os.getenv("FINGENT_LLM_MAX_TOKENS", "4096"))
        self.last_usage: dict = {}   # {prompt_tokens, completion_tokens, total_tokens} from last call

    @property
    def enabled(self) -> bool:
        """True when a model is configured — i.e. the LLM runtime can actually reason."""
        return bool(self.api_key)

    @property
    def name(self) -> str:
        return f"{self.model} @ {self.base_url}"

    def usage_split(self) -> tuple[int, int, bool]:
        """Return (prompt_tokens, completion_tokens, estimated) from the most recent call.
        `estimated` is True only when the provider returned no usage and we approximated."""
        u = self.last_usage or {}
        return (int(u.get("prompt_tokens", 0) or 0),
                int(u.get("completion_tokens", 0) or 0),
                bool(u.get("_estimated")))

    def _body(self, messages, tools, temperature, parallel_tool_calls,
              response_format, tool_choice) -> dict:
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature,
                      "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
            # parallel_tool_calls is an OpenAI extension that some OpenAI-compatible gateways
            # (incl. certain Groq models) reject with a 400. Opt-in only.
            if os.getenv("FINGENT_LLM_PARALLEL_TOOLS", "0") == "1":
                body["parallel_tool_calls"] = parallel_tool_calls
        if response_format:
            body["response_format"] = response_format
        return body

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.1, parallel_tool_calls: bool = False,
             response_format: dict | None = None, tool_choice: str = "auto") -> dict:
        """One chat-completions turn. Returns the assistant message dict, which may contain
        `tool_calls`. Raises on transport / HTTP errors (the caller records the failure)."""
        import requests

        import time as _t

        def _post(tc):
            body = self._body(messages, tools, temperature, parallel_tool_calls,
                              response_format, tc)
            r = None
            for attempt in range(4):
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json=body, timeout=self.timeout,
                )
                # transient rate-limit / gateway errors: back off and retry (respect Retry-After)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    _t.sleep(_retry_after(r, attempt))
                    continue
                break
            r.raise_for_status()
            return r

        try:
            resp = _post(tool_choice)
        except requests.HTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", 0)
            # some gateways reject tool_choice="required" — fall back to "auto" so the agent runs
            if tools and tool_choice not in (None, "auto", "none") and code in (400, 415, 422):
                resp = _post("auto")
            else:
                raise
        data = resp.json()
        self.last_usage = data.get("usage") or {}
        return data["choices"][0]["message"]

    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None,
                    temperature: float = 0.1, parallel_tool_calls: bool = False,
                    tool_choice: str = "auto"):
        """Stream one chat-completions turn as OpenAI-style SSE deltas. Yields dicts:
          {"content": "<token>"}            for assistant content tokens, and/or
          {"tool_calls": [ ... ]}           accumulated tool-call fragments (by index).
        Finally yields {"finish": <reason>, "message": <assembled assistant message>} so the
        caller has the complete tool_calls/content even though they arrived incrementally.
        """
        import requests
        body = self._body(messages, tools, temperature, parallel_tool_calls, None, tool_choice)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}   # OpenAI/Groq: emit a final usage chunk
        self.last_usage = {}
        content_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        finish_reason = None
        import time as _t
        resp = None
        for attempt in range(4):
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=body, timeout=self.timeout, stream=True,
            )
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                delay = _retry_after(resp, attempt)
                resp.close()
                _t.sleep(delay)
                continue
            break
        with resp:
            resp.raise_for_status()
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/event-stream" not in ctype:
                # the gateway ignored stream=True and returned a whole completion — handle it
                # so streaming works against any OpenAI-compatible endpoint (and our test mocks)
                data = resp.json()
                self.last_usage = data.get("usage") or {}
                msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
                if msg.get("content"):
                    yield {"content": msg["content"]}
                fr = (data.get("choices") or [{}])[0].get("finish_reason") or "stop"
                yield {"finish": fr, "message": {
                    "role": "assistant", "content": msg.get("content"),
                    **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})}}
                return
            # SSE responses often omit a charset, so requests defaults to ISO-8859-1 and
            # iter_lines(decode_unicode=True) would mangle multibyte UTF-8 (e.g. an em-dash
            # becomes "â€""). Force UTF-8 so streamed model text is decoded correctly.
            resp.encoding = "utf-8"
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    self.last_usage = chunk["usage"]      # final usage chunk (include_usage)
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    yield {"content": delta["content"]}
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    slot = tool_acc.setdefault(
                        idx, {"id": "", "type": "function",
                              "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
        if not self.last_usage and (content_parts or tool_acc):
            # provider returned no usage on the stream: approximate so the count is HONESTLY
            # labelled estimated (never a flat fabricated constant)
            approx = max(1, len("".join(content_parts)) // 4)
            self.last_usage = {"completion_tokens": approx, "prompt_tokens": 0,
                               "total_tokens": approx, "_estimated": True}
        message: dict = {"role": "assistant", "content": "".join(content_parts) or None}
        if tool_acc:
            message["tool_calls"] = [tool_acc[i] for i in sorted(tool_acc)]
        yield {"finish": finish_reason or "stop", "message": message}
