import os
import re
import json
import requests
from typing import Dict, List, Optional, Tuple, Union

class UniversalLLMClient:
    def __init__(self, api_key: str = "", provider: str = "auto", model: str = ""):
        self.api_key = api_key.strip() if api_key else ""
        self.provider = provider
        self.model = model.strip() if model else ""

    def detect_provider(self, key: str) -> str:
        key = key.strip()
        if key.startswith("sk-or-v1-") or "openrouter" in key.lower():
            return "openrouter"
        elif key.startswith("gsk_"):
            return "groq"
        elif key.startswith("AIza"):
            return "gemini"
        elif key.startswith("sk-ant-"):
            return "anthropic"
        elif key.startswith("sk-"):
            return "openai"
        return "openrouter"

    def get_endpoint_and_headers(self, api_key: str, provider: str) -> Tuple[str, Dict[str, str], str]:
        key = api_key.strip()
        eff_provider = provider if provider != "auto" else self.detect_provider(key)

        if eff_provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "MediCare AI Platform",
                "Content-Type": "application/json",
            }
            default_model = "meta-llama/llama-3.3-70b-instruct"
        elif eff_provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            default_model = "gpt-4o-mini"
        elif eff_provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            default_model = "llama-3.3-70b-versatile"
        elif eff_provider == "gemini":
            url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            default_model = "gemini-1.5-flash"
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            default_model = "meta-llama/llama-3.3-70b-instruct"

        return url, headers, default_model

    def complete(self, messages: List[Dict], api_key: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        key = (api_key or self.api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ValueError("API Key is missing. Please enter your API Key in the sidebar.")

        prov = provider or self.provider or "auto"
        url, headers, default_model = self.get_endpoint_and_headers(key, prov)

        target_model = (model or self.model or default_model).strip()
        clean_model = target_model.replace(":free", "")

        candidate_models = [clean_model, target_model, default_model]
        if prov == "openrouter" or self.detect_provider(key) == "openrouter":
            candidate_models.extend(["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct", "google/gemini-flash-1.5", "openrouter/auto"])
        elif prov == "openai":
            candidate_models.extend(["gpt-4o-mini", "gpt-4o"])
        elif prov == "groq":
            candidate_models.extend(["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])

        last_error = ""
        for mod in list(dict.fromkeys(candidate_models)):
            if not mod:
                continue
            payload = {
                "model": mod,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=90)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    try:
                        last_error = resp.json().get("error", {}).get("message", resp.text)
                    except Exception:
                        last_error = resp.text
            except Exception as e:
                last_error = str(e)

        raise RuntimeError(f"API Provider Error: {last_error}")
