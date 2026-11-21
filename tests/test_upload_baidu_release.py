import hashlib
import io
import json
import urllib.error
import urllib.parse
from email import policy
from email.parser import BytesParser

import pytest

from scripts import upload_baidu_release as uploader


def release(tmp_path, contents):
    archive = tmp_path / "FaceMatching-test.zip"
    archive.write_bytes(contents)
    digest = hashlib.sha256(contents).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


def test_upload_pair_sends_only_missing_parts_and_retries_slice(tmp_path, monkeypatch):
    # Force a multi-part ZIP but keep the checksum as a single part.
    monkeypatch.setattr(uploader, "BLOCK_SIZE", 128)
    monkeypatch.setattr(uploader.time, "sleep", lambda seconds: None)
    archive = release(tmp_path, b"x" * 128 + b"last")
    checksum = archive.with_suffix(".zip.sha256").read_bytes()
    directory = "/apps/测试/Visual-Analysis"
    calls = []
    uploads = {
        f"{directory}/{archive.name}": [b"x" * 128, b"last"],
        f"{directory}/{archive.name}.sha256": [checksum],
    }

    class SimulatedBaidu:
        def open(self, request, timeout):
            url = urllib.parse.urlsplit(request.full_url)
            query = urllib.parse.parse_qs(url.query)
            assert query["access_token"] == ["test-token"]
            method = query["method"][0]
            if method == "upload":
                assert url.hostname == "d.pcs.baidu.com"
                assert query["type"] == ["tmpfile"]
                path = query["path"][0]
                index = int(query["partseq"][0])
                assert query["uploadid"] == ["job"]
                content_type = request.get_header("Content-type")
                message = BytesParser(policy=policy.default).parsebytes(
                    f"Content-Type: {content_type}\r\n\r\n".encode("ascii") + request.data
                )
                part, = message.iter_parts()
                assert part.get_param("name", header="content-disposition") == "file"
                assert part.get_payload(decode=True) == uploads[path][index]
                prior_calls = len(calls)
                calls.append((method, path, index))
                if prior_calls == 1:
                    raise urllib.error.HTTPError(request.full_url, 503, "", {}, None)
                result = {"md5": uploader.block_md5(uploads[path][index])}
            else:
                assert url.hostname == "pan.baidu.com"
                form = urllib.parse.parse_qs(request.data.decode("ascii"))
                path = form["path"][0]
                blocks = uploads[path]
                assert form["rtype"] == ["0"]
                assert json.loads(form["block_list"][0]) == [uploader.block_md5(block) for block in blocks]
                calls.append((method, path, None))
                if method == "precreate":
                    assert form["autoinit"] == ["1"]
                    result = {"errno": 0, "return_type": 1, "uploadid": "job", "block_list": [len(blocks) - 1]}
                else:
                    assert method == "create"
                    assert form["uploadid"] == ["job"]
                    assert int(form["size"][0]) == sum(len(block) for block in blocks)
                    result = {"errno": 0, "path": path, "size": int(form["size"][0])}
            return io.BytesIO(json.dumps(result).encode("utf-8"))

    client = uploader.Netdisk("test-token")
    client.opener = SimulatedBaidu()
    uploader.upload_release(archive, directory, client)
    assert [(method, index) for method, _, index in calls] == [
        ("precreate", None), ("upload", 1), ("upload", 1), ("create", None),
        ("precreate", None), ("upload", 0), ("create", None),
    ]


def test_invalid_archive_is_rejected_before_any_network_request(tmp_path):
    archive = release(tmp_path, b"original")
    archive.write_bytes(b"tampered")
    with pytest.raises(uploader.UploadError, match="does not match"):
        uploader.upload_release(archive, "/apps/approved", None)


def test_rapid_upload_only_uses_precreate(tmp_path):
    archive = release(tmp_path, b"original")
    posted = []

    class SimulatedBaidu:
        def post(self, endpoint, method, *, fields):
            assert method == "precreate"
            posted.append(fields["path"])
            return {"errno": 0, "return_type": 2}

    uploader.upload_release(archive, "/apps/approved", SimulatedBaidu())
    assert posted == [
        f"/apps/approved/{archive.name}",
        f"/apps/approved/{archive.name}.sha256",
    ]
