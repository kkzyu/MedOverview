import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_s: int = 60


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        # Load .env if available so users don't have to rely on setx/restart.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing DEEPSEEK_API_KEY.\n"
                "PowerShell (current session): $env:DEEPSEEK_API_KEY=\"...\"\n"
                "PowerShell (persist): setx DEEPSEEK_API_KEY \"...\"  (then reopen terminal)\n"
                "Or create a .env file with DEEPSEEK_API_KEY=..."
            )
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        return cls(DeepSeekConfig(api_key=api_key, base_url=base_url, model=model))

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
        retries: int = 4,
        retry_backoff_s: float = 2.0,
    ) -> dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.cfg.timeout_s
                )
                if resp.status_code == 429:
                    raise RuntimeError(f"rate limited: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not content:
                    raise RuntimeError(f"empty response: {data}")
                try:
                    import json

                    return json.loads(content)
                except Exception as e:
                    raise RuntimeError(f"non-json content: {content[:200]}") from e
            except Exception as e:
                last_err = e
                if attempt >= retries:
                    break
                time.sleep(retry_backoff_s * (attempt + 1))

        raise RuntimeError(f"DeepSeek request failed: {last_err}")
