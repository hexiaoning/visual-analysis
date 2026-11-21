from fractions import Fraction
from threading import Event
from types import SimpleNamespace

import av
import numpy as np
import pytest

from face_matching.face_video_enhancement import enhance_face_video


def _source(path):
    frame_times = [0, 40, 110, 190]
    pixels = np.tile(np.arange(80, dtype=np.uint8), (60, 1))
    pixels = np.repeat(pixels[:, :, None], 3, axis=2)
    with av.open(str(path), "w", format="matroska") as output:
        video = output.add_stream("ffv1", rate=25)
        video.width, video.height, video.pix_fmt = 80, 60, "bgr0"
        video.codec_context.time_base = Fraction(1, 1000)
        audio = output.add_stream("pcm_s16le", rate=48000)
        audio.layout = "mono"
        sound = av.AudioFrame.from_ndarray(np.full((1, 9600), 421, np.int16), format="s16", layout="mono")
        sound.sample_rate, sound.pts, sound.time_base = 48000, 0, Fraction(1, 48000)
        for packet in audio.encode(sound):
            output.mux(packet)
        for pts in frame_times:
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in video.encode(frame):
                output.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None):
                output.mux(packet)
    return pixels, frame_times


def test_keeps_video_frame_timestamps_and_audio(tmp_path):
    source = tmp_path / "原片.mkv"
    result = tmp_path / "增强.mkv"
    pixels, times = _source(source)
    detector = SimpleNamespace(detect=lambda frame: [SimpleNamespace(bbox=(10, 10, 60, 50))])
    assert enhance_face_video(source, result, detector, 12, Event()) == (4, 4)
    with av.open(str(result)) as output:
        frames = list(output.decode(video=0))
        assert [frame.pts * frame.time_base * 1000 for frame in frames] == times
        image = frames[0].to_ndarray(format="bgr24")
        np.testing.assert_array_equal(image[:10], pixels[:10])
        assert not np.array_equal(image[10:50, 10:60], pixels[10:50, 10:60])
    with av.open(str(result)) as output:
        assert output.streams.audio[0].codec_context.name == "pcm_s16le"
        samples = np.concatenate([frame.to_ndarray() for frame in output.decode(audio=0)], axis=1)
        np.testing.assert_array_equal(samples, np.full((1, 9600), 421, np.int16))
    assert source.is_file()


def test_no_face_and_stop_leave_no_export(tmp_path):
    source = tmp_path / "source.mkv"
    result = tmp_path / "result.mkv"
    _source(source)
    detector = SimpleNamespace(detect=lambda frame: [])
    with pytest.raises(ValueError, match="未找到"):
        enhance_face_video(source, result, detector, 12, Event())
    assert not result.exists()
    stopped = Event()
    stopped.set()
    assert enhance_face_video(source, result, detector, 12, stopped) == (0, 0)
    assert not result.exists()
