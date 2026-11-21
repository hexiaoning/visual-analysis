"""Runtime protocol for a customer-authorized vehicle catalogue gateway."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SOURCES = ("miit", "dongchedi")
SOURCE_NAMES = {"miit": "工信部", "dongchedi": "懂车帝"}
CANDIDATE_FIELDS = ("brand", "series", "model", "year", "trim", "catalogue_id")
_TIMEOUT = 8
_MAX_RESPONSE = 8 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _text(value: object, limit: int = 200) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _https(value: object) -> str:
    url = _text(value, 2048)
    try:
        parsed = urlsplit(url)
        if parsed.scheme == "https" and parsed.hostname and not (parsed.username or parsed.password):
            return url
    except ValueError:
        pass
    return ""


def _public_candidates(items: object) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    return [{key: _text(item.get(key)) for key in CANDIDATE_FIELDS}
            for item in items[:3] if isinstance(item, dict) and _text(item.get("series"))]


def _parse_records(payload: object) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("响应须包含 records 数组")
    records = []
    for item in payload["records"][:12]:
        if not isinstance(item, dict):
            raise ValueError("车型资料格式无效")
        record = {key: _text(item.get(key)) for key in CANDIDATE_FIELDS}
        record["record_id"] = _text(item.get("record_id"))
        record["source_url"] = _https(item.get("source_url"))
        if not record["series"] or not (record["record_id"] or record["source_url"]):
            raise ValueError("车型资料缺少车系或来源")
        pictures = item.get("image_jpeg_base64", [])
        record["image_jpeg_base64"] = [
            picture for picture in pictures[:2]
            if isinstance(picture, str) and len(picture) <= 1024 * 1024
        ] if isinstance(pictures, list) else []
        parameters = item.get("parameters")
        record["parameters"] = {key[:80]: _text(value) for key, value in list(parameters.items())[:12]} if (
            isinstance(parameters, dict)
        ) else {}
        records.append(record)
    return records


def _query(source: str, settings: dict | None, candidates: list[dict[str, str]]) -> dict:
    result = {"source": source, "queried_at": "", "status": "not_configured", "records": []}
    if not isinstance(settings, dict):
        return result
    endpoint = _https(settings.get("url"))
    token = os.environ.get(f"VISUAL_VEHICLE_{source.upper()}_TOKEN")
    if not endpoint or not token:
        return result
    result["queried_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = json.dumps({"schema_version": 1, "source": source, "candidates": candidates},
                      ensure_ascii=False).encode("utf-8")
    try:
        request = Request(endpoint, data=body, method="POST", headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json", "Accept": "application/json",
            "Cache-Control": "no-store, no-cache", "User-Agent": "VisualAnalysis/1",
        })
        with build_opener(_NoRedirect()).open(request, timeout=_TIMEOUT) as response:
            payload = response.read(_MAX_RESPONSE + 1)
        if len(payload) > _MAX_RESPONSE:
            raise ValueError("车型资料响应过大")
        result["records"] = _parse_records(json.loads(payload))
        result["status"] = "ok" if result["records"] else "no_match"
    except HTTPError as exc:
        result["status"] = "unauthorized" if exc.code in (401, 403) else "unavailable"
    except (URLError, TimeoutError, OSError):
        result["status"] = "unavailable"
    except ValueError:
        result["status"] = "invalid_response"
    return result


def query_catalogues(settings: dict, raw_candidates: object) -> list[dict]:
    candidates = _public_candidates(raw_candidates)
    if not candidates:
        return [{"source": source, "queried_at": "", "status": "no_candidate", "records": []}
                for source in SOURCES]
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        futures = [pool.submit(_query, source, settings.get(source), candidates) for source in SOURCES]
        return [future.result() for future in futures]


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    return worker_main(Path(sys.argv[1]), Path(sys.argv[2]))


def worker_main(request_path: Path, response_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = query_catalogues(request["settings"], request["candidates"])
    response_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
