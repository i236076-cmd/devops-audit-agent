import pytest
from openai.types.chat import ChatCompletion

from agent_core import llm
from agent_core.llm import (
    AnthropicBackend, LLMClient, LLMError, LLMResponse, OpenAIBackend, _retry_delay, _retry_wait, create_client,
)

KEYS = ["LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_REASONING_EFFORT"]

HISTORY = [
    {"role": "user", "content": "audit"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "name": "list_files", "arguments": {}, "extra": {"google": {"thought_signature": "SIG"}}},
        {"id": "c2", "name": "git_log", "arguments": {"max_count": 3}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": "files"},
    {"role": "tool", "tool_call_id": "c2", "content": "log"},
]


class HttpError(Exception):
    def __init__(self, status, message=""):
        super().__init__(message)
        self.status_code = status


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system, messages, tools):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(llm.time, "sleep", recorded.append)
    return recorded


def test_openai_conversion_pairs_tool_results_and_keeps_signature():
    messages = OpenAIBackend._convert("system prompt", HISTORY)
    assert messages[0] == {"role": "system", "content": "system prompt"}
    calls = messages[2]["tool_calls"]
    assert calls[0]["extra_content"] == {"google": {"thought_signature": "SIG"}}
    assert "extra_content" not in calls[1]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["c1", "c2"]


def test_anthropic_conversion_merges_consecutive_tool_results():
    messages = AnthropicBackend._convert(HISTORY)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [block["type"] for block in messages[1]["content"]] == ["tool_use", "tool_use"]
    assert [block["type"] for block in messages[2]["content"]] == ["tool_result", "tool_result"]


def test_openai_response_parsing_keeps_signature_and_cleans_tool_names():
    raw = {"id": "x", "object": "chat.completion", "created": 1, "model": "m", "choices": [{
        "index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "repo_browser.list_files", "arguments": "{}"},
             "extra_content": {"google": {"thought_signature": "SIG123"}}},
            {"id": "2", "type": "function", "function": {"name": "read_file", "arguments": "not json"}},
        ]}}]}
    completion = ChatCompletion.model_validate(raw)

    class Completions:
        @staticmethod
        def create(**kwargs):
            return completion

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    response = OpenAIBackend(Client(), "model").complete("s", [{"role": "user", "content": "hi"}], [])
    assert [c.name for c in response.tool_calls] == ["list_files", "read_file"]
    assert response.tool_calls[0].extra == {"google": {"thought_signature": "SIG123"}}
    assert response.tool_calls[1].arguments == {}


@pytest.mark.parametrize("text, expected", [
    ("Please retry in 15h50m0.5s.", 15 * 3600 + 50 * 60 + 0.5),
    ("Please try again in 1m23.5s.", 83.5),
    ("Please try again in 850ms.", 0.85),
    ("nothing useful here", None),
])
def test_retry_delay_parsing(text, expected):
    assert _retry_delay(text) == pytest.approx(expected) if expected is not None else _retry_delay(text) is None


def test_retry_wait_decisions():
    assert _retry_wait(HttpError(429, "tokens per minute. Please try again in 2.5s."), 1) == pytest.approx(3.5)
    assert _retry_wait(HttpError(429, "per day limit. Please retry in 15h50m0s."), 1) is None
    assert _retry_wait(HttpError(429, "requests per day exceeded"), 1) is None
    assert _retry_wait(HttpError(429, "slow down"), 1) == 4
    assert _retry_wait(HttpError(503, "busy"), 2) == 4
    assert _retry_wait(HttpError(401, "bad key"), 1) is None
    assert _retry_wait(ValueError("bug"), 1) is None


def test_client_retries_transient_errors_then_succeeds(sleeps):
    backend = ScriptedBackend([HttpError(429, "Please try again in 2s."), HttpError(503, "busy"), LLMResponse("ok")])
    assert LLMClient(backend, "openai", "m").chat("s", []).text == "ok"
    assert backend.calls == 3
    assert len(sleeps) == 2


def test_client_gives_up_after_max_retries(sleeps):
    backend = ScriptedBackend([HttpError(503, "busy")] * 10)
    with pytest.raises(LLMError):
        LLMClient(backend, "openai", "m").chat("s", [])
    assert backend.calls == llm.MAX_RETRIES


def test_client_fails_fast_on_daily_quota(sleeps):
    backend = ScriptedBackend([HttpError(429, "per day limit exceeded. Please retry in 15h50m0s."), LLMResponse("ok")])
    with pytest.raises(LLMError, match="quota"):
        LLMClient(backend, "openai", "gemini-x").chat("s", [])
    assert backend.calls == 1
    assert sleeps == []


def test_auth_errors_do_not_echo_the_key(sleeps):
    backend = ScriptedBackend([HttpError(401, "invalid key sk-" + "x" * 30)])
    with pytest.raises(LLMError) as error:
        LLMClient(backend, "openai", "m").chat("s", [])
    assert "sk-" not in str(error.value)


def test_create_client_requires_configuration(clean_env):
    with pytest.raises(LLMError, match="No LLM configured"):
        create_client()
    clean_env.setenv("OPENAI_API_KEY", "your_openai_key_here")
    with pytest.raises(LLMError):
        create_client()
    clean_env.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        create_client()


def test_create_client_picks_provider_from_available_key(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 20)
    client = create_client()
    assert (client.provider, client.model) == ("anthropic", llm.DEFAULT_MODELS["anthropic"])


def test_create_client_supports_openai_compatible_services(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openai")
    clean_env.setenv("OPENAI_API_KEY", "k" * 20)
    clean_env.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    clean_env.setenv("LLM_MODEL", "my-model")
    clean_env.setenv("LLM_REASONING_EFFORT", "low")
    client = create_client()
    assert client.model == "my-model"
    assert client.backend.reasoning_effort == "low"
    assert "api.example.com" in str(client.backend.client.base_url)
