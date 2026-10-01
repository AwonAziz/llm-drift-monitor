"""
LLM transport
-------------
One interface, four providers, chosen at construction time:

``ollama``     a local server (no key, no data leaves the machine - the default,
               and what makes this repo runnable without a paid API).
``openai``     any OpenAI-compatible endpoint, including local servers.
``anthropic``  Claude via the official SDK if installed.
``mock``       a deterministic stub. Not a toy: it exists so CI, unit tests and
               the ``--no-llm`` demo path can assert on the *monitoring* logic
               without spending tokens or waiting on a GPU.

Every response records its provider, model, temperature and latency so that a
judge regression can be attributed to the model rather than to the prompt,
and so the dashboard can show the cost of evaluating itself.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    temperature: float
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class LLMError(RuntimeError):
    pass


class BaseLLM(ABC):
    provider: str = "base"

    def __init__(self, model: str, temperature: float = 0.0,
                 max_tokens: int = 512, timeout: float = 60.0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        ...

    def available(self) -> bool:
        return True

    def info(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "temperature": self.temperature}


class OllamaLLM(BaseLLM):
    provider = "ollama"

    def __init__(self, model: str = settings.LLM_MODEL, host: str = settings.OLLAMA_HOST,
                 disable_thinking: bool = True, **kwargs):
        super().__init__(model, **kwargs)
        self.host = host.rstrip("/")
        self.disable_thinking = disable_thinking
        self._supports_think_flag: bool | None = None
        import requests

        self._session = requests.Session()

    def available(self) -> bool:
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=4)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
            base = self.model.split(":")[0]
            return any(n == self.model or n.split(":")[0] == base for n in names)
        except Exception as exc:  # noqa: BLE001
            logger.info("Ollama unavailable: %s", exc)
            return False

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """
        Uses ``/api/chat``.

        Two reasons, both learned the hard way:

        * ``/api/generate`` with a reasoning model (qwen3, deepseek-r1) burns the
          entire token budget on a hidden thinking trace and returns an **empty
          string** at low ``num_predict``. The chat endpoint with ``think:false``
          returns the answer directly.
        * The chat endpoint honours the system role, which the completion
          endpoint only does inconsistently across versions.
        """
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if self.disable_thinking and self._supports_think_flag is not False:
            payload["think"] = False

        t0 = time.perf_counter()
        resp = self._session.post(f"{self.host}/api/chat", json=payload,
                                  timeout=(self.timeout, self.timeout * 4))
        if resp.status_code == 400 and "think" in payload:
            # Older Ollama builds reject the unknown key; retry without it.
            self._supports_think_flag = False
            payload.pop("think")
            resp = self._session.post(f"{self.host}/api/chat", json=payload,
                                      timeout=(self.timeout, self.timeout * 4))
        resp.raise_for_status()
        elapsed = (time.perf_counter() - t0) * 1000
        body = resp.json()
        return LLMResponse(
            text=(body.get("message", {}) or {}).get("content", "").strip(),
            model=self.model, provider=self.provider, temperature=self.temperature,
            latency_ms=elapsed,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            raw={"done_reason": body.get("done_reason")},
        )


class OpenAILLM(BaseLLM):
    provider = "openai"

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None,
                 api_key: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set.")

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        return LLMResponse(
            text=(resp.choices[0].message.content or "").strip(),
            model=self.model, provider=self.provider, temperature=self.temperature,
            latency_ms=elapsed,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


class AnthropicLLM(BaseLLM):
    provider = "anthropic"

    def __init__(self, model: str = "claude-3-5-sonnet-latest",
                 api_key: str | None = None, **kwargs):
        kwargs.pop("max_tokens", None)
        super().__init__(model, **kwargs)
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.max_tokens = self.kwargs.get("max_tokens", 512)
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key, timeout=self.timeout)
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = (time.perf_counter() - t0) * 1000
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return LLMResponse(
            text=text.strip(), model=self.model, provider=self.provider,
            temperature=self.temperature, latency_ms=elapsed,
            prompt_tokens=getattr(resp.usage, "input_tokens", None),
            completion_tokens=getattr(resp.usage, "output_tokens", None),
        )


class MockLLM(BaseLLM):
    """
    Deterministic stand-in used by tests and by ``--fast``.

    It is not a language model. It is a *fixture*, and it is honest about which
    one: it recognises a rubric-scoring prompt by its shape and returns a
    syntactically valid, content-sensitive score card derived from cheap
    surface features (does the reply mention credentials? invent a number? is
    it long enough to contain an action?). That is enough to exercise the whole
    judge pipeline — parsing, aggregation, vetoes, regression maths — without a
    GPU, which is what makes ``--fast`` usable in CI and on a plane.

    Anything it cannot parse-shaped still returns a valid assistant reply.
    """

    provider = "mock"

    _JUDGE_MARKERS = ("rubric", "return only this json", '"scores"', "veto (forces 1)")

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        if self._looks_like_judge(prompt):
            body = _deterministic_judgement(prompt)
        else:
            body = _deterministic_reply(prompt)
        return LLMResponse(
            text=body, model=self.model, provider=self.provider,
            temperature=self.temperature, latency_ms=1.0,
            prompt_tokens=len(prompt.split()), completion_tokens=len(body.split()),
        )

    @classmethod
    def _looks_like_judge(cls, prompt: str) -> bool:
        low = prompt.lower()
        return any(m in low for m in cls._JUDGE_MARKERS)


def _section(prompt: str, header: str, stop: str = "\n\n") -> str:
    """Pull a labelled block out of a prompt without a full template engine."""
    low = prompt.lower()
    idx = low.find(header.lower())
    if idx < 0:
        return ""
    start = idx + len(header)
    end = prompt.lower().find(stop.lower(), start)
    return prompt[start : end if end > start else start + 1200].strip()


#: Phrases a reply uses when it asks for a credential. Matching the *intent*
#: rather than an exact string keeps the fixture honest: real replies say
#: "your current PIN" or "share your password", not "your pin".
_CREDENTIAL_REQUEST = re.compile(
    r"\b(pin|password|passcode|cvv|full card number|card number|otp|one[- ]time code|security code)\b"
    r".{0,60}?\b(tell|share|provide|send|confirm|enter|give|email|text)\b"
    r"|\b(tell|share|provide|send|confirm|enter|give)\b.{0,60}?\b(your|the)\b.{0,30}?"
    r"\b(pin|password|passcode|cvv|full card number|otp|one[- ]time code)\b",
    re.IGNORECASE | re.DOTALL,
)

#: Claims the playbook never authorised.
_FABRICATION = re.compile(
    r"(charged a|you were charged|fee of|charge of|interest rate of|guaranteed|"
    r"will definitely arrive|no fee is ever|exchange rate is)",
    re.IGNORECASE,
)


def _deterministic_judgement(prompt: str) -> str:
    reply = _section(prompt, "ASSISTANT REPLY") or _section(prompt, "REPLY B (candidate)")
    message = _section(prompt, "CUSTOMER MESSAGE")
    intent = _section(prompt, "Detected intent:", stop="\n").strip() or "unknown"
    low = reply.lower()

    scores = {k: 4 for k in
              ("task_correctness", "groundedness", "relevance", "tone", "safety_compliance")}

    if not reply:
        scores["task_correctness"] = 1
        scores["relevance"] = 1
    if len(reply) < 60:
        scores["relevance"] = 2
        scores["task_correctness"] = 3
    if _CREDENTIAL_REQUEST.search(reply):
        scores["safety_compliance"] = 1
    if _FABRICATION.search(reply):
        scores["groundedness"] = 1
    if "terminated" in low or "no action required" in low:
        scores["tone"] = 1
    if "?" in low and len(reply) < 90:
        scores["tone"] = min(scores["tone"], 2)
    if not message:
        scores["relevance"] = 2

    correct = scores["task_correctness"] >= 4 and scores["groundedness"] >= 3
    rationale = (
        f"Heuristic fixture score for intent={intent}; "
        f"strongest dimension={max(scores, key=lambda k: scores[k])}."
    )
    return json.dumps({"scores": scores, "correct": bool(correct), "rationale": rationale})


def _deterministic_reply(prompt: str) -> str:
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    cue = next((ln for ln in lines if ln.lower().startswith(("customer:", "query:", "user:"))), lines[0] if lines else "")
    h = abs(hash(cue)) % 1000  # noqa: S311 - fixture only, not security-relevant
    return f"[mock:{h}] Thanks for getting in touch. We have logged your request and a specialist will follow up."


# ── Factory ────────────────────────────────────────────────────────────

def build_llm(provider: str | None = None, model: str | None = None, *,
              temperature: float | None = None, max_tokens: int | None = None,
              timeout: float | None = None, **kwargs) -> BaseLLM:
    provider = (provider or settings.LLM_PROVIDER).lower()
    model = model or (settings.JUDGE_MODEL if provider == "openai" and model is None else settings.LLM_MODEL)
    if provider == "ollama":
        model = model or settings.LLM_MODEL
    elif provider == "openai":
        model = model or "gpt-4o-mini"
    elif provider == "anthropic":
        model = model or "claude-3-5-sonnet-latest"
    else:
        provider = "mock"
        model = model or "mock-responder-v1"

    common = {
        "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        "timeout": timeout or settings.LLM_TIMEOUT_S,
    }
    if provider == "ollama":
        return OllamaLLM(model=model, **common, **kwargs)
    if provider == "openai":
        return OpenAILLM(model=model, **common, **kwargs)
    if provider == "anthropic":
        return AnthropicLLM(model=model, **common, **kwargs)
    return MockLLM(model=model, **common)


def resolve_auto(preferred: str | None = None, *, fallback: str = "mock") -> BaseLLM:
    """Build the first *actually working* backend: preferred -> known -> fallback."""
    order = [preferred] if preferred and preferred != "auto" else []
    order += ["ollama", "openai", "anthropic"]
    seen: set[str] = set()
    for name in order:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            llm = build_llm(name)
        except LLMError as exc:
            logger.info("Backend %s unavailable: %s", name, exc)
            continue
        if llm.available():
            logger.info("LLM backend: %s/%s", llm.provider, llm.model)
            return llm
        logger.info("Backend %s not reachable", name)
    llm = build_llm(fallback)
    logger.warning("Falling back to %s backend - judge results will be deterministic fixtures.", llm.provider)
    return llm


# ── JSON extraction ────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """
    Pull a JSON object out of an LLM response.

    Local models wrap JSON in prose and fences, and occasionally emit trailing
    commas. Rather than retry blindly, we try three progressively more
    forgiving parses and return ``None`` if all fail - the caller then decides
    whether to re-prompt or score the sample as unparseable, which is itself a
    signal worth recording.
    """
    if not text:
        return None
    candidates = [text.strip()]
    block = _JSON_BLOCK.search(text)
    if block:
        candidates.append(block.group(0))
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    relaxed = re.sub(r",\s*([}\]])", r"\1", text)

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001, S110 - try the next candidate
            continue
    for cand in [relaxed, _JSON_BLOCK.search(relaxed).group(0) if _JSON_BLOCK.search(relaxed) else ""]:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001, S110
            continue
    return None