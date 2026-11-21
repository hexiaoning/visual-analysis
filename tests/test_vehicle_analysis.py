from datetime import datetime
import json
from urllib.error import URLError

import numpy as np
import pytest

from face_matching import vehicle_provider
from face_matching.vehicle_analysis import (
    VehicleCandidate, VehicleMatch, _compare_online, _run_catalogue_worker, load_catalogue,
)
from face_matching.vision.local_media import local_path, open_local_video


@pytest.mark.parametrize("path", ["https://example.test/clip.mp4", "rtsp://camera/live",
                                  r"\\server\share\photo.jpg", "//server/share/photo.jpg"])
def test_visual_media_rejects_remote_paths(path):
    with pytest.raises(ValueError):
        local_path(path)


def test_video_suffix_cannot_hide_remote_playlist(tmp_path):
    fake_video = tmp_path / "playlist.mp4"
    fake_video.write_bytes(b"#EXTM3U\nhttps://example.test/video.ts\n")
    with pytest.raises(ValueError, match="容器"):
        open_local_video(fake_video)


def test_catalogue_only_allows_local_reference_images(tmp_path):
    catalogue = tmp_path / "vehicles.json"
    catalogue.write_text(json.dumps({"schema_version": 1, "vehicles": [
        {"brand": "示例", "series": "A", "images": ["../outside.jpg"]},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="参考库目录"):
        load_catalogue(catalogue)


def test_each_recognition_queries_both_sources_and_sends_only_catalogue_fields(monkeypatch):
    monkeypatch.setenv("VISUAL_VEHICLE_MIIT_TOKEN", "test-token")
    monkeypatch.setenv("VISUAL_VEHICLE_DONGCHEDI_TOKEN", "test-token")
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return json.dumps({"records": [
                {"brand": "示例", "series": "A", "record_id": "source-1",
                 "source_url": "https://example.test/record"},
            ]}).encode()

    class Opener:
        def open(self, request, *, timeout):
            requests.append((request.full_url, json.loads(request.data)))
            return Response()

    monkeypatch.setattr(vehicle_provider, "build_opener", lambda *args: Opener())
    settings = {source: {"url": f"https://example.test/{source}"} for source in vehicle_provider.SOURCES}
    query = [{"brand": "示例", "series": "A", "plate": "private", "image_bytes": "private"}]
    for _ in range(2):
        results = vehicle_provider.query_catalogues(settings, query)
        assert [result["status"] for result in results] == ["ok", "ok"]
        assert all(datetime.fromisoformat(result["queried_at"]).tzinfo for result in results)
        assert all(result["records"][0]["record_id"] == "source-1" for result in results)
    assert len(requests) == 4
    assert {url for url, _ in requests} == {item["url"] for item in settings.values()}
    assert all("plate" not in payload["candidates"][0] and "image_bytes" not in payload["candidates"][0]
               for _, payload in requests)


def test_offline_and_unconfigured_sources_never_count_as_verified(monkeypatch):
    monkeypatch.setenv("VISUAL_VEHICLE_MIIT_TOKEN", "test-token")
    monkeypatch.delenv("VISUAL_VEHICLE_DONGCHEDI_TOKEN", raising=False)

    class Unavailable:
        def open(self, *args, **kwargs):
            raise URLError("offline")

    monkeypatch.setattr(vehicle_provider, "build_opener", lambda *args: Unavailable())
    settings = {source: {"url": f"https://example.test/{source}"} for source in vehicle_provider.SOURCES}
    results = vehicle_provider.query_catalogues(settings, [{"brand": "示例", "series": "A"}])
    assert results[0]["queried_at"]
    assert [result["status"] for result in results] == ["unavailable", "not_configured"]
    assert _compare_online(results, [VehicleMatch(VehicleCandidate("示例", "A"), 0.9)], [], None) == (
        "本地初判，未联网核验"
    )


def test_launches_real_catalogue_worker_with_unconfigured_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    monkeypatch.delenv("VISUAL_VEHICLE_MIIT_TOKEN", raising=False)
    monkeypatch.delenv("VISUAL_VEHICLE_DONGCHEDI_TOKEN", raising=False)
    (tmp_path / "vehicle-providers.json").write_text("{}", encoding="utf-8")
    result = _run_catalogue_worker([VehicleCandidate("示例", "A")])
    assert [item["source"] for item in result] == ["miit", "dongchedi"]
    assert [item["status"] for item in result] == ["not_configured", "not_configured"]


def test_one_online_document_preserves_source_and_local_match_level():
    record = {key: "" for key in vehicle_provider.CANDIDATE_FIELDS}
    record.update(brand="示例", series="A", model="M", year="2025", record_id="example")
    results = [
        {"status": "ok", "records": [record]},
        {"status": "unavailable", "records": []},
    ]
    matches = [VehicleMatch(VehicleCandidate("示例", "A", "M", "2026"), 0.95)]
    assert _compare_online(results, matches, [np.array([1.0])], None) == "部分联网核验"
    assert record["candidate_levels"] == {"示例 / A / M / 2026": "车系"}
    assert record["record_id"] == "example"

def test_inconsistent_online_years_are_shown_without_confirming_a_trim():
    first = {key: "" for key in vehicle_provider.CANDIDATE_FIELDS}
    first.update(brand="示例", series="A", model="M", year="2025", record_id="one")
    second = dict(first, year="2026", record_id="two")
    results = [{"status": "ok", "records": [record]} for record in (first, second)]
    matches = [VehicleMatch(VehicleCandidate("示例", "A"), 0.95)]
    assert _compare_online(results, matches, [], None) == "已查双源 · 车款资料有差异，请复核"
    assert "工信部 2025 · 懂车帝 2026" in results[0]["differences"][0]


def test_reachable_catalogues_without_results_are_distinct_from_offline():
    results = [{"status": "no_match", "records": []} for _ in vehicle_provider.SOURCES]
    assert _compare_online(results, [], [], None) == "已查在线资料 · 未找到一致车型"
