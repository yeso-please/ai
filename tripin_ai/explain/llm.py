"""LLM 호출부. 공급자를 바꿔 끼울 수 있게 `generate_json`만 약속한다.

지금은 Gemini(데모 레포와 같은 `gemini-flash-lite-latest`). 키는 환경변수 GEMINI_API_KEY 또는
GEMINI_API_KEY_FILE(YAML의 gemini.api.key)에서 읽고, URL이 아니라 요청 헤더로 보낸다(URL은 로그에 남기 쉽다).

⚠️ 무료 등급은 입력이 제품 개선에 쓰일 수 있다. 회원 설문 답·메모는 LLM에 보내지 않는다(prompts.py).
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"


class LLMUnavailable(Exception):
    """키 없음·한도 초과·네트워크 오류·형식 오류. 호출부는 규칙 문장으로 대체한다."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class LLMClient(Protocol):
    model: str

    def generate_json(self, system: str, user: str) -> dict: ...


def load_gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key and os.environ.get("GEMINI_API_KEY_FILE"):
        text = Path(os.environ["GEMINI_API_KEY_FILE"]).read_text(encoding="utf-8")
        match = re.search(r"^gemini:\s*\n(?:[ \t]+.*\n)*?[ \t]+key:\s*['\"]?([^'\"\s]+)", text, re.M)
        key = match.group(1) if match else None
    return key or None


class GeminiClient:
    def __init__(self, key: str, model: str = DEFAULT_GEMINI_MODEL, timeout: int = 30, temperature: float = 0.9):
        self.key, self.model, self.timeout, self.temperature = key, model, timeout, temperature

    def generate_json(self, system: str, user: str) -> dict:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"},
        }
        request = urllib.request.Request(
            GEMINI_URL.format(model=self.model), data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMUnavailable("RATE_LIMITED" if e.code == 429 else f"HTTP_{e.code}")
        except (urllib.error.URLError, TimeoutError):
            raise LLMUnavailable("NETWORK")
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (KeyError, IndexError, json.JSONDecodeError):
            raise LLMUnavailable("MALFORMED_RESPONSE")


def default_client() -> GeminiClient | None:
    key = load_gemini_key()
    return GeminiClient(key, os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)) if key else None
