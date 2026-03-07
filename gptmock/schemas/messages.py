from __future__ import annotations

import base64
from typing import Any


def _normalize_image_data_url(url: str) -> str:
    try:
        if not isinstance(url, str):
            return url
        if not url.startswith("data:image/"):
            return url
        if ";base64," not in url:
            return url
        header, data = url.split(",", 1)
        try:
            from urllib.parse import unquote

            data = unquote(data)
        except Exception:
            pass
        data = data.strip().replace("\n", "").replace("\r", "")
        data = data.replace("-", "+").replace("_", "/")
        pad = (-len(data)) % 4
        if pad:
            data = data + ("=" * pad)
        try:
            base64.b64decode(data, validate=True)
        except Exception:
            return url
        return f"{header},{data}"
    except Exception:
        return url


def _convert_tool_message(message: dict[str, Any]) -> dict[str, Any] | None:
    call_id = message.get("tool_call_id") or message.get("id")
    if not isinstance(call_id, str) or not call_id:
        return None
    content = message.get("content", "")
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str) and text:
                    texts.append(text)
        content = "\n".join(texts)
    if not isinstance(content, str):
        return None
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": content,
    }


def _convert_assistant_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        tc_type = tc.get("type", "function")
        if tc_type != "function":
            continue
        call_id = tc.get("id") or tc.get("call_id")
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(call_id, str) and isinstance(name, str) and isinstance(args, str):
            out.append(
                {
                    "type": "function_call",
                    "name": name,
                    "arguments": args,
                    "call_id": call_id,
                },
            )
    return out


def _convert_content_parts(content: Any, role: Any) -> list[dict[str, Any]]:
    content_items: list[dict[str, Any]] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str) and text:
                    kind = "output_text" if role == "assistant" else "input_text"
                    content_items.append({"type": kind, "text": text})
            elif ptype == "image_url":
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                if isinstance(url, str) and url:
                    content_items.append(
                        {
                            "type": "input_image",
                            "image_url": _normalize_image_data_url(url),
                        },
                    )
    elif isinstance(content, str) and content:
        kind = "output_text" if role == "assistant" else "input_text"
        content_items.append({"type": kind, "text": content})
    return content_items


def convert_chat_messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue

        if role == "tool":
            tool_item = _convert_tool_message(message)
            if tool_item:
                input_items.append(tool_item)
            continue

        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            input_items.extend(_convert_assistant_tool_calls(message))

        content_items = _convert_content_parts(message.get("content", ""), role)

        if not content_items:
            continue
        role_out = "assistant" if role == "assistant" else "user"
        input_items.append(
            {"type": "message", "role": role_out, "content": content_items},
        )
    return input_items


def convert_tools_chat_to_responses(tools: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            continue
        desc = fn.get("description") if isinstance(fn, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "name": name,
                "description": desc or "",
                "strict": False,
                "parameters": params,
            },
        )
    return out
