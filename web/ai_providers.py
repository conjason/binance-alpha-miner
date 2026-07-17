"""AI provider resolution for training analysis (SAFE fork).

安全说明（相对上游 BinanceAlphaMiner 的改动）
------------------------------------------------
上游此文件包含一段从本地其它软件（腾讯 CodeBuddy / WorkBuddy Electron 桌面端）
读取并用 Windows DPAPI + AES-GCM 解密登录会话 token 的代码（`_decrypt_electron_token`
等），手法与浏览器凭证窃取木马一致。本 fork **完整移除** 该逻辑。

本文件只支持「用户显式提供 API Key」的通道，绝不读取、解密或扫描本机上任何其它
应用的凭证/会话/存储：

  - deepseek:  官方 DeepSeek，模型固定，base_url=https://api.deepseek.com
  - openai_compatible:  任意 OpenAI 兼容端点（用户填 Key，端点用环境变量或默认）

Key 只来自 Web 界面输入框或用户自设的环境变量，直接作为 Bearer 发往对应服务商。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# OpenAI 兼容通道：默认模型/端点可用环境变量覆盖，Key 始终由用户提供
_OAI_DEFAULT_MODEL = os.environ.get("OPENAI_COMPAT_MODEL", "gpt-4o-mini")
_OAI_DEFAULT_BASE_URL = os.environ.get("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1")

PROVIDERS = ("deepseek", "openai_compatible")


@dataclass
class ResolvedProvider:
    provider: str
    model: str
    base_url: str
    api_key: str
    label: str
    needs_user_key: bool = True


def provider_status() -> dict[str, Any]:
    """返回可用 AI 通道列表。全部通道都需要用户自带 Key（needs_user_key=True）。

    本 fork 不做任何本地 token 自动探测，故不存在「已自动读取登录 token」这类状态。
    """
    return {
        "providers": [
            {
                "id": "deepseek",
                "label": "DeepSeek (deepseek-v4-flash)",
                "available": True,
                "needs_user_key": True,
                "hint": "固定模型 deepseek-v4-flash · https://api.deepseek.com · 需自填 API Key",
            },
            {
                "id": "openai_compatible",
                "label": "OpenAI 兼容端点",
                "available": True,
                "needs_user_key": True,
                "hint": (
                    f"任意 OpenAI 兼容 /chat/completions 端点（默认 {_OAI_DEFAULT_BASE_URL}，"
                    "可用环境变量 OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_MODEL 覆盖）· 需自填 API Key"
                ),
            },
        ]
    }


def resolve_provider(provider: str, api_key: str | None = None) -> ResolvedProvider:
    """把通道 id + 用户 Key 解析为具体的调用参数。

    与上游最大的差别：**没有任何** 从本地会话/DPAPI/Electron 存储自动取 token 的分支，
    Key 必须由调用方（Web 界面输入或环境变量）显式传入。
    """
    pid = (provider or "deepseek").strip().lower()
    key = (api_key or "").strip()

    if pid not in PROVIDERS:
        raise ValueError(f"不支持的 AI 通道: {provider}（可选: {', '.join(PROVIDERS)}）")

    if pid == "deepseek":
        if not key:
            key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise ValueError("请填写 DeepSeek API Key（或设置环境变量 DEEPSEEK_API_KEY）")
        return ResolvedProvider(
            provider="deepseek",
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=key,
            label="DeepSeek",
            needs_user_key=True,
        )

    # openai_compatible
    if not key:
        key = os.environ.get("OPENAI_COMPAT_API_KEY", "").strip() or os.environ.get(
            "OPENAI_API_KEY", ""
        ).strip()
    if not key:
        raise ValueError(
            "请填写 API Key（或设置环境变量 OPENAI_COMPAT_API_KEY / OPENAI_API_KEY）"
        )
    return ResolvedProvider(
        provider="openai_compatible",
        model=_OAI_DEFAULT_MODEL,
        base_url=_OAI_DEFAULT_BASE_URL,
        api_key=key,
        label="OpenAI 兼容端点",
        needs_user_key=True,
    )


def chat_completions(
    resolved: ResolvedProvider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> str:
    """Call OpenAI-compatible /chat/completions and return assistant text."""
    parts: list[str] = []
    for chunk in stream_chat_completions(
        resolved, messages, max_tokens=max_tokens, timeout=timeout
    ):
        parts.append(chunk)
    content = "".join(parts).strip()
    if not content:
        raise RuntimeError("AI 返回内容为空")
    return content


def stream_chat_completions(
    resolved: ResolvedProvider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    timeout: float = 180.0,
):
    """Yield text deltas from OpenAI-compatible streaming chat completions.

    只向 `resolved.base_url`（deepseek 官方或用户自配的 OpenAI 兼容端点）发送请求，
    Authorization 用的是用户显式提供的 Key，不涉及任何本机其它应用的凭证。
    """
    import urllib.error
    import urllib.request

    url = resolved.base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": resolved.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {resolved.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "AlphaForge-AI-Analyze",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if not text:
                    text = delta.get("reasoning_content") or ""
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"AI 请求失败 HTTP {exc.code}: {err_body}") from exc
    except Exception as exc:
        raise RuntimeError(f"AI 请求失败: {exc}") from exc
