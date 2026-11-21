"""Upload a verified release pair to the Baidu Netdisk Open Platform."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import BinaryIO

BLOCK_SIZE = 4 * 1024 * 1024
MAX_FILE_SIZE = 1024 * BLOCK_SIZE
FILE_API = "https://pan.baidu.com/rest/2.0/xpan/file"
SLICE_API = "https://d.pcs.baidu.com/rest/2.0/pcs/superfile2"


class UploadError(Exception):
    pass


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def block_md5(block: bytes) -> str:
    return hashlib.md5(block, usedforsecurity=False).hexdigest()


class Netdisk:
    def __init__(self, token: str):
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirects())

    def post(
        self,
        endpoint: str,
        method: str,
        *,
        params: dict | None = None,
        fields: dict | None = None,
        block: bytes | None = None,
    ) -> dict:
        query = urllib.parse.urlencode({
            "method": method, "access_token": self.token, **(params or {}),
        })
        if block is None:
            body = urllib.parse.urlencode(fields or {}).encode("ascii")
            content_type = "application/x-www-form-urlencoded"
        else:
            boundary = uuid.uuid4().hex
            content_type = f"multipart/form-data; boundary={boundary}"
            header = (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="file"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("ascii")
            body = header + block + f"\r\n--{boundary}--\r\n".encode("ascii")

        request = urllib.request.Request(
            f"{endpoint}?{query}", data=body, headers={"Content-Type": content_type},
        )
        # Retry a slice with the same upload ID and index. Avoid retrying a
        # create request automatically when its response may have been lost.
        attempts = 3 if block is not None else 1
        for attempt in range(attempts):
            try:
                with self.opener.open(request, timeout=90) as response:
                    result = json.loads(response.read(256 * 1024))
                break
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if attempt + 1 == attempts or status not in (429, 500, 502, 503, 504):
                    raise UploadError(f"{method}: HTTP {status}") from None
            except (urllib.error.URLError, OSError):
                if attempt + 1 == attempts:
                    raise UploadError(f"{method}: connection failed or timed out") from None
            except (ValueError, UnicodeError):
                raise UploadError(f"{method}: invalid Baidu response") from None
            time.sleep(2**attempt)

        if not isinstance(result, dict):
            raise UploadError(f"{method}: invalid Baidu response")
        errno = result.get("errno")
        # The slice API can signal success with only an MD5 and request ID.
        if errno == 0 or (method == "upload" and errno is None and "md5" in result):
            return result
        if type(errno) is int:
            raise UploadError(f"{method}: Baidu returned errno {errno}")
        raise UploadError(f"{method}: incomplete Baidu response")


def upload_file(
    client: Netdisk, stream: BinaryIO, remote_path: str, size: int, expected_sha256: str | None = None,
) -> None:
    if not 0 < size <= MAX_FILE_SIZE:
        raise UploadError("file size must be between 1 byte and 4 GiB")
    digest = hashlib.sha256()
    part_hashes = []
    count = 0
    for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
        count += len(block)
        part_hashes.append(block_md5(block))
        digest.update(block)
    if count != size or (expected_sha256 and digest.hexdigest() != expected_sha256):
        raise UploadError("the local ZIP does not match its SHA-256 file")

    metadata = {
        "path": remote_path, "size": size, "isdir": 0, "rtype": 0,
        "block_list": json.dumps(part_hashes, separators=(",", ":")),
    }
    initial = client.post(FILE_API, "precreate", fields={**metadata, "autoinit": 1})
    if initial.get("return_type") == 2:
        return
    upload_id = initial.get("uploadid")
    missing = initial.get("block_list")
    if not isinstance(upload_id, str) or not upload_id or not isinstance(missing, list):
        raise UploadError("precreate: missing upload ID or part list")
    for index in missing:
        if type(index) is not int or not 0 <= index < len(part_hashes):
            raise UploadError("precreate: invalid part index")
        stream.seek(index * BLOCK_SIZE)
        block = stream.read(BLOCK_SIZE)
        if block_md5(block) != part_hashes[index]:
            raise UploadError("the local file changed during upload")
        uploaded = client.post(
            SLICE_API, "upload", block=block,
            params={"type": "tmpfile", "path": remote_path, "uploadid": upload_id, "partseq": index},
        )
        if uploaded.get("md5") != part_hashes[index]:
            raise UploadError(f"upload: Baidu returned an unexpected MD5 for part {index}")

    created = client.post(FILE_API, "create", fields={**metadata, "uploadid": upload_id})
    if created.get("path") != remote_path or created.get("size") != size:
        raise UploadError("create: Baidu did not confirm the expected file path and size")


def upload_release(archive: Path, directory: str, client: Netdisk) -> None:
    parts = directory.split("/")
    if (len(parts) < 3 or parts[:2] != ["", "apps"]
            or any(part in ("", ".", "..") for part in parts[2:])
            or any(char in directory for char in '\\:*?"<>|')
            or any(ord(char) < 32 for char in directory)):
        raise UploadError("BAIDU_NETDISK_RELEASE_DIR must be an absolute /apps/<app> path")
    if archive.suffix.lower() != ".zip":
        raise UploadError("expected a release ZIP")
    checksum_file = archive.with_suffix(archive.suffix + ".sha256")
    with checksum_file.open("rb") as checksum:
        checksum_bytes = checksum.read(4096)
    fields = checksum_bytes.decode("utf-8").strip().split(maxsplit=1)
    if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]) or fields[1] != archive.name:
        raise UploadError("invalid release SHA-256 file")

    with archive.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        upload_file(client, stream, f"{directory}/{archive.name}", size, fields[0].lower())
    print(f"Baidu Netdisk confirmed: {directory}/{archive.name}", flush=True)
    upload_file(client, io.BytesIO(checksum_bytes), f"{directory}/{checksum_file.name}", len(checksum_bytes))
    print(f"Baidu Netdisk confirmed: {directory}/{checksum_file.name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    token = os.environ.get("BAIDU_NETDISK_ACCESS_TOKEN", "").strip()
    directory = os.environ.get("BAIDU_NETDISK_RELEASE_DIR", "").strip().rstrip("/")
    if not token or not directory:
        parser.exit(1, "Configure BAIDU_NETDISK_ACCESS_TOKEN and BAIDU_NETDISK_RELEASE_DIR first.\n")
    try:
        upload_release(args.archive, directory, Netdisk(token))
    except (UploadError, OSError, UnicodeError) as error:
        parser.exit(1, f"Baidu Netdisk upload failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
