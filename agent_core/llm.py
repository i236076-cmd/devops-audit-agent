"""LLM access layer for the auditing agent.

The agent loop talks to one small interface (LLMClient.chat) and never sees
provider details. Two backends are supported:

  * openai     - OpenAI, or any OpenAI-compatible API via LLM_BASE_URL
  * anthropic  - Anthropic Claude

Internal message format (provider neutral):
  {"role": "user", "content": "text"}
  {"role": "assistant", "content": "text", "tool_calls": [{"id", "name", "arguments"}]}
  {"role": "tool", "tool_call_id": "...", "content": "tool output"}
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from tools import redact_secrets

load_dotenv()

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
}
MAX_RETRIES = 4
MAX_RATE_WAIT = 60  # never sleep longer than this for a rate limit (seconds)
REQUEST_TIMEOUT = 90  # seconds
MAX_OUTPUT_TOKENS = 4096


class LLMError(Exception):
    """Raised when the LLM cannot be reached or configured."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    extra: dict = None  # provider metadata that must be echoed back (Gemini thought signature)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)


# --------------------------------------------------------------------------
# OpenAI (and OpenAI-compatible) backend
# --------------------------------------------------------------------------
class OpenAIBackend:
    def __init__(self, client, model, reasoning_effort=None):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def _convert(system, messages):
        out = [{"role": "system", "content": system}]
        for msg in messages:
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                tool_calls = []
                for call in msg["tool_calls"]:
                    entry = {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    if call.get("extra"):  # Gemini 3 needs its thought signature sent back
                        entry["extra_content"] = call["extra"]
                    tool_calls.append(entry)
                out.append({
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": tool_calls,
                })
            elif msg["role"] == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"],
                })
            else:
                out.append({"role": msg["role"], "content": msg["content"]})
        return out

    def complete(self, system, messages, tools, force_tool=None):
        kwargs = {"model": self.model, "messages": self._convert(system, messages)}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if tools:
            kwargs["tools"] = [{"type": "function", "function": tool} for tool in tools]
        if force_tool:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            extra = getattr(call, "extra_content", None)
            if extra is None:
                extra = (getattr(call, "model_extra", None) or {}).get("extra_content")
            # some models prefix tool names (e.g. repo_browser.list_files)
            name = call.function.name.split(".")[-1]
            calls.append(ToolCall(call.id, name, arguments, extra))
        return LLMResponse(message.content or "", calls)


# --------------------------------------------------------------------------
# Anthropic backend
# --------------------------------------------------------------------------
class AnthropicBackend:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    @staticmethod
    def _convert(messages):
        out = []
        for msg in messages:
            if msg["role"] == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
                # several tool results in a row must share one user message
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif msg["role"] == "assistant":
                blocks = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for call in msg.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["arguments"],
                    })
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "user", "content": msg["content"]})
        return out

    def complete(self, system, messages, tools, force_tool=None):
        kwargs = {
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": system,
            "messages": self._convert(messages),
        }
        if tools:
            kwargs["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
                for tool in tools
            ]
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}
        response = self.client.messages.create(**kwargs)
        text_parts, calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input or {})))
        return LLMResponse("".join(text_parts), calls)


# --------------------------------------------------------------------------
# Public client with retries
# --------------------------------------------------------------------------
def _is_parse_failure(text):
    """The provider rejected a tool call the model wrote badly (a model glitch, not our bug)."""
    return bool(re.search(r"output_parse_failed|Parsing failed|failed_generation", text))


def _is_retryable(exc):
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504, 529):
        return True
    if status == 400 and _is_parse_failure(str(exc)):
        return True  # sampling again usually produces a valid tool call
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


def _retry_delay(text):
    """Seconds the server asked us to wait ("retry in 1m23.5s", "try again in 850ms")."""
    match = re.search(r"(?:try again|retry) in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", text)
    if not match:
        return None
    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(
        float(value) * units[unit]
        for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", match.group(1))
    )


def _retry_wait(exc, attempt):
    """Seconds to wait before retrying, or None when retrying is pointless."""
    if not _is_retryable(exc):
        return None
    if getattr(exc, "status_code", None) == 400:
        return 1  # malformed model output: just ask again
    if getattr(exc, "status_code", None) == 429:
        text = str(exc)
        delay = _retry_delay(text)
        if delay is None:
            if re.search(r"per day|perday|daily", text, re.IGNORECASE):
                return None  # daily quota: waiting a few seconds will not help
            return 2 ** attempt * 2
        return None if delay > MAX_RATE_WAIT else delay + 1
    return 2 ** attempt


class LLMClient:
    def __init__(self, backend, provider, model):
        self.backend = backend
        self.provider = provider
        self.model = model

    def chat(self, system, messages, tools=None, force_tool=None):
        """Send one request and return an LLMResponse. Retries transient errors."""
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.backend.complete(system, messages, tools or [], force_tool)
            except Exception as exc:  # SDK exception types differ per provider
                last_error = exc
                wait = _retry_wait(exc, attempt)
                if wait is None or attempt == MAX_RETRIES:
                    break
                print(f"[llm] {type(exc).__name__}, retrying in {wait:.0f}s "
                      f"(attempt {attempt}/{MAX_RETRIES})", file=sys.stderr)
                time.sleep(wait)
        status = getattr(last_error, "status_code", None)
        if status in (401, 403):
            raise LLMError(
                f"{self.provider} rejected the API key (HTTP {status}). Check your .env file."
            )
        if status == 429:
            delay = _retry_delay(str(last_error))
            when = f" Retry in about {int(delay)}s." if delay else ""
            raise LLMError(
                f"{self.provider} rate limit or quota reached for {self.model}.{when} "
                "Wait, or change LLM_MODEL / the provider in .env."
            )
        detail = redact_secrets(str(last_error))[:300]
        raise LLMError(f"{self.provider} request failed: {type(last_error).__name__}: {detail}")


def _real_key(name):
    """Return the env var value, ignoring empty or placeholder values."""
    value = (os.getenv(name) or "").strip()
    if not value or value.lower().startswith("your_"):
        return None
    return value


def create_client():
    """Build an LLMClient from environment variables (.env supported)."""
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    openai_key = _real_key("OPENAI_API_KEY")
    anthropic_key = _real_key("ANTHROPIC_API_KEY")

    if not provider:
        provider = "openai" if openai_key else "anthropic" if anthropic_key else ""
    if provider not in DEFAULT_MODELS:
        raise LLMError(
            "No LLM configured. Copy .env.example to .env and set LLM_PROVIDER "
            "(openai or anthropic) plus the matching API key."
        )

    model = (os.getenv("LLM_MODEL") or "").strip() or DEFAULT_MODELS[provider]

    if provider == "openai":
        if not openai_key:
            raise LLMError("OPENAI_API_KEY is missing or still a placeholder in .env")
        from openai import OpenAI

        client = OpenAI(
            api_key=openai_key,
            base_url=(os.getenv("LLM_BASE_URL") or "").strip() or None,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )
        effort = (os.getenv("LLM_REASONING_EFFORT") or "").strip() or None
        return LLMClient(OpenAIBackend(client, model, effort), provider, model)

    if not anthropic_key:
        raise LLMError("ANTHROPIC_API_KEY is missing or still a placeholder in .env")
    from anthropic import Anthropic

    client = Anthropic(api_key=anthropic_key, timeout=REQUEST_TIMEOUT, max_retries=0)
    return LLMClient(AnthropicBackend(client, model), provider, model)
