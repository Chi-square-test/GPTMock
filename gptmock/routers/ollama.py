from __future__ import annotations

import datetime
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from gptmock.core.dependencies import get_http_client, get_settings
from gptmock.core.logging import log_json
from gptmock.core.settings import Settings
from gptmock.schemas.requests import OllamaChatRequest, OllamaShowRequest
from gptmock.schemas.transform import convert_ollama_messages, normalize_ollama_tools
from gptmock.services.chat import ChatCompletionError, process_chat_completion
from gptmock.services.model_registry import OLLAMA_FAKE_EVAL, get_ollama_models

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_openai_payload(ollama_payload: dict[str, Any], model: str) -> dict[str, Any]:
    raw_messages = ollama_payload.get("messages")
    messages = convert_ollama_messages(
        raw_messages,
        ollama_payload.get("images")
        if isinstance(ollama_payload.get("images"), list)
        else None,
    )

    if isinstance(messages, list):
        sys_idx = next(
            (
                i
                for i, m in enumerate(messages)
                if isinstance(m, dict) and m.get("role") == "system"
            ),
            None,
        )
        if isinstance(sys_idx, int):
            sys_msg = messages.pop(sys_idx)
            content = sys_msg.get("content") if isinstance(sys_msg, dict) else ""
            messages.insert(0, {"role": "user", "content": content})

    stream_req = ollama_payload.get("stream")
    if stream_req is None:
        stream_req = True
    stream_req = bool(stream_req)

    openai_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream_req,
    }

    tools_req = (
        ollama_payload.get("tools")
        if isinstance(ollama_payload.get("tools"), list)
        else []
    )
    if tools_req:
        openai_tools = normalize_ollama_tools(tools_req)
        if openai_tools:
            openai_payload["tools"] = openai_tools

    tool_choice = ollama_payload.get("tool_choice", "auto")
    if tool_choice in ("auto", "none"):
        openai_payload["tool_choice"] = tool_choice

    parallel_tool_calls = bool(ollama_payload.get("parallel_tool_calls", False))
    if parallel_tool_calls:
        openai_payload["parallel_tool_calls"] = parallel_tool_calls

    responses_tools_payload = (
        ollama_payload.get("responses_tools")
        if isinstance(ollama_payload.get("responses_tools"), list)
        else []
    )
    if responses_tools_payload:
        openai_payload["responses_tools"] = responses_tools_payload

    responses_tool_choice = ollama_payload.get("responses_tool_choice")
    if isinstance(responses_tool_choice, str) and responses_tool_choice in (
        "auto",
        "none",
    ):
        openai_payload["responses_tool_choice"] = responses_tool_choice

    return openai_payload


async def _convert_openai_to_ollama_stream(
    response: Any, model: str,
) -> AsyncGenerator[bytes]:
    try:
        async for sse_chunk in response:
            if not sse_chunk.startswith(b"data: "):
                continue

            json_bytes = sse_chunk[6:].strip()

            if json_bytes == b"[DONE]":
                done_chunk = {
                    "model": model,
                    "created_at": datetime.datetime.now(
                        datetime.UTC,
                    ).isoformat()
                    + "Z",
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    **OLLAMA_FAKE_EVAL,
                }
                yield (json.dumps(done_chunk) + "\n").encode("utf-8")
                break

            try:
                openai_chunk = json.loads(json_bytes)
                choices = openai_chunk.get("choices", [])

                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        ollama_chunk = {
                            "model": model,
                            "created_at": datetime.datetime.now(
                                datetime.UTC,
                            ).isoformat()
                            + "Z",
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                            "done": False,
                        }
                        yield (json.dumps(ollama_chunk) + "\n").encode("utf-8")
            except Exception:
                logger.debug("Failed to parse OpenAI SSE chunk JSON", exc_info=True)
                continue
    finally:
        if hasattr(response, "aclose"):
            await response.aclose()


def _convert_openai_to_ollama_response(
    response: dict[str, Any], model: str,
) -> dict[str, Any]:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})

    return {
        "model": model,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        "message": message,
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
        **OLLAMA_FAKE_EVAL,
    }


@router.get("/api/version")
async def ollama_version(
    settings: Settings = Depends(get_settings),
):
    """Return Ollama version."""
    if settings.verbose:
        logger.debug("IN GET /api/version")

    version = settings.ollama_version
    payload = {"version": version}

    if settings.verbose:
        log_json("OUT GET /api/version", payload, logger=logger.debug)

    return JSONResponse(payload)


@router.get("/api/tags")
async def ollama_tags(
    settings: Settings = Depends(get_settings),
):
    """List available models in Ollama format."""
    if settings.verbose:
        logger.debug("IN GET /api/tags")

    models = get_ollama_models(expose_reasoning=settings.expose_reasoning_models)

    payload = {"models": models}

    if settings.verbose:
        log_json("OUT GET /api/tags", payload, logger=logger.debug)

    return JSONResponse(payload)


@router.post("/api/show")
async def ollama_show(
    body: OllamaShowRequest,
    settings: Settings = Depends(get_settings),
):
    """Show model details."""
    if settings.verbose:
        log_json("IN POST /api/show", body.model_dump(), logger=logger.debug)

    if not body.model.strip():
        err = {"error": "Model not found"}
        if settings.verbose:
            log_json("OUT POST /api/show", err, logger=logger.debug)
        return JSONResponse(err, status_code=400)

    # Return hardcoded model info (from routes_ollama.py:184-202)
    response = {
        "modelfile": '# Modelfile generated by "ollama show"\n# To build a new Modelfile based on this one, replace the FROM line with:\n# FROM llava:latest\n\nFROM /models/blobs/sha256:placeholder\nTEMPLATE """{{ .System }}\nUSER: {{ .Prompt }}\nASSISTANT: """\nPARAMETER num_ctx 100000\nPARAMETER stop "</s>"\nPARAMETER stop "USER:"\nPARAMETER stop "ASSISTANT:"',
        "parameters": 'num_keep 24\nstop "<|start_header_id|>"\nstop "<|end_header_id|>"\nstop "<|eot_id|>"',
        "template": "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "8.0B",
            "quantization_level": "Q4_0",
        },
        "model_info": {
            "general.architecture": "llama",
            "general.file_type": 2,
            "llama.context_length": 2000000,
        },
        "capabilities": ["completion", "vision", "tools", "thinking"],
    }

    if settings.verbose:
        log_json("OUT POST /api/show", response, logger=logger.debug)

    return JSONResponse(response)


@router.post("/api/chat")
async def ollama_chat(
    body: OllamaChatRequest,
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    """Ollama-compatible chat endpoint.

    Converts Ollama format → OpenAI format → calls service → converts back to Ollama ndjson.
    """
    ollama_payload = body.model_dump()
    model = body.model

    if settings.verbose:
        log_json("IN POST /api/chat", ollama_payload, logger=logger.debug)

    raw_messages = ollama_payload.get("messages")

    if not isinstance(raw_messages, list) or not raw_messages:
        err = {"error": "Invalid request format"}
        if settings.verbose:
            log_json("OUT POST /api/chat", err, logger=logger.debug)
        return JSONResponse(err, status_code=400)

    openai_payload = _build_openai_payload(ollama_payload, model)

    # 3. Call service layer
    try:
        response, is_streaming = await process_chat_completion(
            payload=openai_payload,
            settings=settings,
            http_client=http_client,
        )

        # 4. Convert response to Ollama format
        if is_streaming:
            if settings.verbose:
                logger.debug("OUT POST /api/chat (streaming response)")

            return StreamingResponse(
                _convert_openai_to_ollama_stream(response, model),
                media_type="application/x-ndjson",
            )
        ollama_response = _convert_openai_to_ollama_response(response, model)

        if settings.verbose:
            log_json("OUT POST /api/chat", ollama_response, logger=logger.debug)

        return JSONResponse(ollama_response)

    except ChatCompletionError as e:
        error_response = {"error": e.message}
        if settings.verbose:
            log_json("OUT POST /api/chat ERROR", error_response, logger=logger.debug)
        return JSONResponse(error_response, status_code=e.status_code)
