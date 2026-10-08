from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any

import requests


ENDPOINTS = {
    "openai-compatible": "https://dashscope.aliyuncs.com/compatible-mode/v1/images/generations",
    "dashscope": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
}
PROTOCOL_LABELS = {
    "openai-compatible": "Alibaba Cloud Model Studio OpenAI-compatible Images",
    "dashscope": "Alibaba Cloud Model Studio DashScope synchronous multimodal generation",
}
MODEL = "qwen-image-3.0-pro"
SEED = 42
INPUT_ROLES = (
    "edit_target_roi_rgb",
    "projected_control_overlay",
    "same_track_temporal_anchor",
)

CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 600
DOWNLOAD_READ_TIMEOUT_SECONDS = 120
MAX_INPUT_IMAGE_BYTES = 10 * 1024 * 1024
MAX_OUTPUT_IMAGE_BYTES = 50 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class TrialError(Exception):
    def __init__(self, code: str, status: str = "failed") -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _png_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 24 or not raw.startswith(PNG_SIGNATURE):
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return width, height


def _jpeg_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        return None
    offset = 2
    standalone_markers = {0x01, *range(0xD0, 0xD9)}
    sof_markers = {
        *range(0xC0, 0xC4),
        *range(0xC5, 0xC8),
        *range(0xC9, 0xCC),
        *range(0xCD, 0xD0),
    }
    while offset + 1 < len(raw):
        if raw[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(raw) and raw[offset] == 0xFF:
            offset += 1
        if offset >= len(raw):
            return None
        marker = raw[offset]
        offset += 1
        if marker in standalone_markers:
            continue
        if offset + 2 > len(raw):
            return None
        segment_length = int.from_bytes(raw[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(raw):
            return None
        if marker in sof_markers:
            if segment_length < 7:
                return None
            height = int.from_bytes(raw[offset + 3 : offset + 5], "big")
            width = int.from_bytes(raw[offset + 5 : offset + 7], "big")
            return width, height
        offset += segment_length
    return None


def _load_data_uri(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TrialError("input_read_failed") from exc
    if not raw:
        raise TrialError("input_empty")
    if len(raw) > MAX_INPUT_IMAGE_BYTES:
        raise TrialError("input_too_large")

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
        dimensions = _jpeg_dimensions(raw)
    elif suffix == ".png":
        media_type = "image/png"
        dimensions = _png_dimensions(raw)
    else:
        raise TrialError("unsupported_input_format")
    if dimensions is None or dimensions[0] <= 0 or dimensions[1] <= 0:
        raise TrialError("invalid_input_image")

    encoded = base64.b64encode(raw).decode("ascii")
    metadata = {
        "file_name": path.name,
        "media_type": media_type,
        "byte_count": len(raw),
        "width": dimensions[0],
        "height": dimensions[1],
    }
    return f"data:{media_type};base64,{encoded}", metadata


def _parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", value)
    if match is None:
        raise TrialError("invalid_size_format")
    width, height = int(match.group(1)), int(match.group(2))
    pixel_area = width * height
    if pixel_area < 512 * 512 or pixel_area > 2048 * 2048:
        raise TrialError("size_pixel_area_out_of_range")
    if max(width / height, height / width) > 8:
        raise TrialError("size_aspect_ratio_out_of_range")
    return width, height


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Qwen Image 3.0 Pro edit against the official Beijing endpoint."
    )
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=tuple(ENDPOINTS),
        default="openai-compatible",
        help="Wire protocol; the original OpenAI-compatible path remains the default.",
    )
    parser.add_argument(
        "--size",
        default="1024x576",
        help="Requested output size as WIDTHxHEIGHT; inputs are never resized locally.",
    )
    args = parser.parse_args()
    if not 1 <= len(args.images) <= 3:
        parser.error("--images requires between 1 and 3 paths")
    return args


def _redact_message(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+\b", "[REDACTED_KEY]", value)
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+",
        r"\1[REDACTED_KEY]",
        text,
    )
    text = re.sub(
        r"data:[^;,\s]+;base64,[A-Za-z0-9+/=]+",
        "[REDACTED_DATA_URI]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"https?://[^\s\"'<>]+",
        lambda match: match.group(0).split("?", 1)[0]
        + ("?[REDACTED_QUERY]" if "?" in match.group(0) else ""),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?is)(request[_\s-]*body|request[_\s-]*payload)\s*[:=].*$",
        r"\1=[REDACTED_BODY]",
        text,
    )
    text = " ".join(text.split())
    return text[:160]


def _safe_remote_error(body: Any, status: int) -> tuple[str, str]:
    remote_code: Any = None
    remote_message: Any = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            remote_code = error.get("code")
            remote_message = error.get("message")
        remote_code = remote_code or body.get("code")
        remote_message = remote_message or body.get("message")

    if isinstance(remote_code, str):
        sanitized_code = re.sub(r"[^A-Za-z0-9_.:-]", "_", remote_code)[:96]
    else:
        sanitized_code = ""
    if sanitized_code:
        return sanitized_code, _redact_message(remote_message)

    if status in {401, 403}:
        code = "auth_or_permission_denied"
    elif status == 404:
        code = "endpoint_or_model_not_found"
    elif status == 400:
        code = "invalid_request"
    elif status == 429:
        code = "rate_limited"
    elif 500 <= status <= 599:
        code = "server_error"
    else:
        code = f"http_{status}"
    return code, _redact_message(remote_message)


def _model_needs_user(code: str, message: str) -> bool:
    code_lower = code.lower().replace("-", "_").replace(".", "_")
    explicit_codes = {
        "invalidmodel",
        "invalid_model",
        "modelnotfound",
        "model_not_found",
        "modelnotavailable",
        "model_not_available",
        "modelunsupported",
        "model_unsupported",
        "modelnotexist",
        "model_not_exist",
        "modelaccessdenied",
        "model_access_denied",
        "modelpermissiondenied",
        "model_permission_denied",
    }
    if code_lower in explicit_codes:
        return True
    message_lower = message.lower()
    if MODEL.lower() not in message_lower:
        return False
    return any(
        phrase in message_lower
        for phrase in (
            "not available",
            "not supported",
            "does not exist",
            "not exist",
            "not found",
            "unsupported model",
            "do not have access",
            "no access",
            "no permission",
            "当前区域",
            "不支持",
            "不可用",
            "不存在",
        )
    )


def _write_attempt(
    path: Path,
    *,
    protocol: str,
    endpoint: str,
    http_status: int | None,
    status: str,
    error_code: str | None,
    message: str,
) -> None:
    _write_json_atomic(
        path,
        {
            "protocol": protocol,
            "endpoint": endpoint,
            "http_status": http_status,
            "status": status,
            "error": {
                "code": error_code,
                "message": _redact_message(message),
            },
        },
    )


def _extract_usage(body: dict[str, Any]) -> dict[str, Any]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return {}

    allowed_fields = {
        "output_width",
        "output_height",
        "input_image_count",
        "input_image_type",
        "output_image_count",
        "output_image_type",
    }
    return {key: usage[key] for key in sorted(allowed_fields) if key in usage}


def _post_once(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
) -> requests.Response:
    try:
        return session.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
    except requests.ConnectTimeout as exc:
        raise TrialError("connect_timeout") from exc
    except requests.ReadTimeout as exc:
        raise TrialError("read_timeout") from exc
    except requests.ConnectionError as exc:
        raise TrialError("connection_error") from exc
    except requests.RequestException as exc:
        raise TrialError("request_error") from exc


def _download_png_once(session: requests.Session, url: str) -> bytes:
    if not url.startswith("https://"):
        raise TrialError("invalid_result_url")
    try:
        response = session.get(
            url,
            timeout=(CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS),
            allow_redirects=True,
        )
    except requests.ConnectTimeout as exc:
        raise TrialError("download_connect_timeout") from exc
    except requests.ReadTimeout as exc:
        raise TrialError("download_read_timeout") from exc
    except requests.ConnectionError as exc:
        raise TrialError("download_connection_error") from exc
    except requests.RequestException as exc:
        raise TrialError("download_request_error") from exc

    if response.status_code != 200:
        raise TrialError(f"download_http_{response.status_code}")
    content = response.content
    if len(content) > MAX_OUTPUT_IMAGE_BYTES:
        raise TrialError("download_too_large")
    if not content.startswith(PNG_SIGNATURE):
        raise TrialError("download_not_png")
    return content


def _build_payload(
    protocol: str,
    prompt: str,
    data_uris: list[str],
    size: str,
) -> dict[str, Any]:
    parameters = {
        "n": 1,
        "seed": SEED,
        "prompt_extend": False,
        "enable_thinking": False,
        "watermark": False,
    }
    if protocol == "openai-compatible":
        return {
            "model": MODEL,
            "prompt": prompt,
            "image": data_uris,
            "size": size,
            **parameters,
        }
    if protocol == "dashscope":
        content = [{"image": data_uri} for data_uri in data_uris]
        content.append({"text": prompt})
        return {
            "model": MODEL,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": {
                **parameters,
                "size": size.replace("x", "*", 1),
            },
        }
    raise TrialError("unsupported_protocol")


def _extract_result_url(protocol: str, body: dict[str, Any]) -> str:
    if protocol == "openai-compatible":
        data = body.get("data")
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise TrialError("invalid_result_count")
        result_url = data[0].get("url")
        if not isinstance(result_url, str) or not result_url:
            raise TrialError("missing_result_url")
        return result_url

    output = body.get("output")
    if not isinstance(output, dict):
        raise TrialError("invalid_response_shape")
    choices = output.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise TrialError("invalid_result_count")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TrialError("invalid_response_shape")
    content = message.get("content")
    if not isinstance(content, list):
        raise TrialError("invalid_response_shape")
    image_urls = [
        item.get("image")
        for item in content
        if isinstance(item, dict) and isinstance(item.get("image"), str) and item.get("image")
    ]
    if len(image_urls) != 1:
        raise TrialError("invalid_result_count")
    return image_urls[0]


def _run(args: argparse.Namespace) -> None:
    output_dir = args.outdir.resolve()
    candidate_path = output_dir / "candidate.png"
    request_metadata_path = output_dir / "request_metadata.json"
    attempt_path = output_dir / "attempt.json"
    usage_path = output_dir / "usage.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if candidate_path.exists():
        raise TrialError("output_exists")

    try:
        prompt_bytes = args.prompt_file.read_bytes()
    except OSError as exc:
        raise TrialError("prompt_read_failed") from exc
    if not prompt_bytes or len(prompt_bytes) > 256 * 1024:
        raise TrialError("invalid_prompt_file")
    try:
        prompt = prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrialError("prompt_not_utf8") from exc
    if not prompt.strip():
        raise TrialError("empty_prompt")

    output_width, output_height = _parse_size(args.size)

    data_uris: list[str] = []
    inputs_metadata: list[dict[str, Any]] = []
    for index, path in enumerate(args.images):
        data_uri, metadata = _load_data_uri(path)
        data_uris.append(data_uri)
        inputs_metadata.append({"role": INPUT_ROLES[index], **metadata})

    target_width = inputs_metadata[0]["width"]
    target_height = inputs_metadata[0]["height"]
    if target_width * output_height != target_height * output_width:
        raise TrialError("size_aspect_mismatch")

    endpoint = ENDPOINTS[args.protocol]
    wire_size = args.size if args.protocol == "openai-compatible" else args.size.replace("x", "*", 1)
    request_metadata = {
        "endpoint": endpoint,
        "region": "China (Beijing)",
        "protocol": PROTOCOL_LABELS[args.protocol],
        "model": MODEL,
        "prompt_file_name": args.prompt_file.name,
        "prompt_byte_count": len(prompt_bytes),
        "parameters": {
            "n": 1,
            "size": wire_size,
            "seed": SEED,
            "prompt_extend": False,
            "enable_thinking": False,
            "watermark": False,
        },
        "inputs": inputs_metadata,
        "redactions": ["api_key", "authorization_header", "input_data_uris", "result_signed_url"],
    }
    _write_json_atomic(request_metadata_path, request_metadata)

    payload = _build_payload(args.protocol, prompt, data_uris, args.size)

    api_key = getpass.getpass("DashScope API key (hidden): ")
    if not api_key:
        raise TrialError("missing_api_key")

    with requests.Session() as session:
        session.trust_env = False
        try:
            response = _post_once(session, endpoint, api_key, payload)
        except TrialError as exc:
            _write_attempt(
                attempt_path,
                protocol=args.protocol,
                endpoint=endpoint,
                http_status=None,
                status=exc.status,
                error_code=exc.code,
                message="",
            )
            raise
        finally:
            del api_key

        try:
            body = response.json()
        except (ValueError, requests.JSONDecodeError) as exc:
            if response.status_code != 200:
                code, message = _safe_remote_error(None, response.status_code)
                _write_attempt(
                    attempt_path,
                    protocol=args.protocol,
                    endpoint=endpoint,
                    http_status=response.status_code,
                    status="failed",
                    error_code=code,
                    message=message,
                )
                raise TrialError(code) from exc
            _write_attempt(
                attempt_path,
                protocol=args.protocol,
                endpoint=endpoint,
                http_status=response.status_code,
                status="failed",
                error_code="invalid_json_response",
                message="",
            )
            raise TrialError("invalid_json_response") from exc

        if response.status_code != 200:
            code, message = _safe_remote_error(body, response.status_code)
            status = "needs_user" if _model_needs_user(code, message) else "failed"
            _write_attempt(
                attempt_path,
                protocol=args.protocol,
                endpoint=endpoint,
                http_status=response.status_code,
                status=status,
                error_code=code,
                message=message,
            )
            raise TrialError(code, status=status)
        if not isinstance(body, dict):
            _write_attempt(
                attempt_path,
                protocol=args.protocol,
                endpoint=endpoint,
                http_status=response.status_code,
                status="failed",
                error_code="invalid_response_shape",
                message="",
            )
            raise TrialError("invalid_response_shape")

        try:
            result_url = _extract_result_url(args.protocol, body)
        except TrialError as exc:
            code, message = _safe_remote_error(body, response.status_code)
            if code == "http_200":
                code = exc.code
            status = "needs_user" if _model_needs_user(code, message) else "failed"
            _write_attempt(
                attempt_path,
                protocol=args.protocol,
                endpoint=endpoint,
                http_status=response.status_code,
                status=status,
                error_code=code,
                message=message,
            )
            raise TrialError(code, status=status) from exc

        _write_attempt(
            attempt_path,
            protocol=args.protocol,
            endpoint=endpoint,
            http_status=response.status_code,
            status="success",
            error_code=None,
            message="",
        )

        image_bytes = _download_png_once(session, result_url)

    output_dimensions = _png_dimensions(image_bytes)
    if output_dimensions != (output_width, output_height):
        raise TrialError("output_size_mismatch")

    usage = {
        "http_status": response.status_code,
        "candidate_byte_count": len(image_bytes),
        "usage": _extract_usage(body),
    }
    request_id = body.get("request_id") if args.protocol == "dashscope" else None
    if not isinstance(request_id, str) or not request_id:
        request_id = response.headers.get("x-request-id")
    if request_id:
        usage["request_id"] = request_id

    _write_bytes_atomic(candidate_path, image_bytes)
    _write_json_atomic(usage_path, usage)


def main() -> int:
    args = _parse_args()
    try:
        _run(args)
    except TrialError as exc:
        print(f"status={exc.status} error_code={exc.code}")
        return 1
    except (OSError, TypeError, ValueError):
        print("status=failed error_code=local_processing_error")
        return 1
    except Exception:
        print("status=failed error_code=unexpected_local_error")
        return 1
    except KeyboardInterrupt:
        print("status=failed error_code=interrupted")
        return 130

    print("status=success error_code=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
