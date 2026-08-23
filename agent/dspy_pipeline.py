"""
PreSense DSPy 코칭 파이프라인.

구조:
  StressAnalysis (입력) + RAG 논문 (근거)
    → ChainOfThought (원인 추론)
    → 구체적 코칭 피드백 (출력)

Luxia API 클라이언트 포함 (환경변수: LUXIA_API_KEY)
"""

from __future__ import annotations
import os
import requests
import dspy


# ══════════════════════════════════════════════════════════════════════════════
#  Luxia API 클라이언트
# ══════════════════════════════════════════════════════════════════════════════

LUXIA_URL   = "https://bridge.luxiacloud.com/luxia/v1/chat"
LUXIA_MODEL = "luxia3-llm-32b-0731"


class LuxiaClient:
    def __init__(self, api_key: str | None = None, max_tokens: int = 300,
                 timeout: int = 30):
        self.api_key    = api_key or os.getenv("LUXIA_API_KEY", "")
        self.max_tokens = max_tokens
        self.timeout    = timeout
        self.messages   = _MessagesNamespace(self)

    def complete(self, messages: list[dict], system: str | None = None) -> str:
        if system:
            messages = [{"role": "system", "content": system}] + messages
        resp = requests.post(
            LUXIA_URL,
            headers={"apikey": self.api_key, "Content-Type": "application/json"},
            json={"model": LUXIA_MODEL, "messages": messages, "max_tokens": self.max_tokens},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class _MessagesNamespace:
    def __init__(self, client: LuxiaClient):
        self._client = client

    def create(self, messages: list[dict], system: str | None = None,
               model: str | None = None, max_tokens: int | None = None,
               **kwargs) -> "_Response":
        if max_tokens:
            self._client.max_tokens = max_tokens
        return _Response(self._client.complete(messages, system=system))


class _Response:
    def __init__(self, text: str):
        self.content = [_Content(text)]


class _Content:
    def __init__(self, text: str):
        self.text = text


class LuxiaLM:
    """DSPy 3.x configure(lm=...) 용 Luxia 어댑터."""

    def __init__(self, api_key: str | None = None, max_tokens: int = 300):
        self._client = LuxiaClient(api_key=api_key, max_tokens=max_tokens)
        self.history: list[dict] = []
        self.kwargs  = {"max_tokens": max_tokens}
        self.model   = LUXIA_MODEL
        self.cache   = False

    def __call__(self, prompt: str | None = None,
                 messages: list[dict] | None = None, **kwargs) -> list[str]:
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        try:
            text = self._client.complete(messages)
        except Exception as e:
            print(f"[LuxiaLM] API 오류: {e}")
            text = ""
        self.history.append({"messages": messages, "response": text})
        return [text]

    def copy(self, **kwargs) -> "LuxiaLM":
        return LuxiaLM(api_key=self._client.api_key,
                       max_tokens=kwargs.get("max_tokens", self._client.max_tokens))

    def dump_state(self): return {"model": self.model, "kwargs": self.kwargs}
    def load_state(self, state): pass


# ══════════════════════════════════════════════════════════════════════════════
#  DSPy 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

class StressCoachSignature(dspy.Signature):
    """
    발표 스트레스 코칭 전문가.
    멀티모달 스트레스 분석 결과와 연구 근거를 바탕으로
    발표자에게 즉시 실천 가능한 2~3문장의 한국어 피드백을 제공합니다.
    과학적 근거를 자연스럽게 녹이되, 전문 용어는 쓰지 마세요.
    """
    stress_analysis:   str = dspy.InputField(desc="현재 스트레스 분석 요약 (점수, 주도 채널, CBM 개념, 관찰된 특징)")
    research_context:  str = dspy.InputField(desc="관련 연구 근거 (논문 발췌)")
    previous_feedback: str = dspy.InputField(desc="이전 피드백 목록 (중복 방지용)")

    feedback: str = dspy.OutputField(desc="즉시 실천 가능한 2~3문장 코칭 피드백 (한국어)")


class StressCoachModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.coach = dspy.ChainOfThought(StressCoachSignature)

    def forward(self, stress_analysis: str, research_context: str,
                previous_feedback: str = "없음") -> dspy.Prediction:
        return self.coach(
            stress_analysis=stress_analysis,
            research_context=research_context,
            previous_feedback=previous_feedback,
        )


def init_dspy(api_key: str | None = None) -> bool:
    """DSPy LM 설정 (솔트룩스 Luxia 사용)."""
    key = api_key or os.environ.get("LUXIA_API_KEY")
    if not key:
        return False
    try:
        dspy.configure(lm=LuxiaLM(api_key=key, max_tokens=200))
        return True
    except Exception as e:
        print(f"⚠️  DSPy LM 설정 실패: {e}")
        return False
