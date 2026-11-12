# Copyright 2026 Torbjörn E. M. Nordling <t@nordlinglab.org>
# SPDX-License-Identifier: Apache-2.0

"""Video metadata, checksum, and frame decoding through FFmpeg.

Frames are decoded as stored, without display rotation (data/schema.md §1),
and each frame comes with its presentation timestamp in the stream time base.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


def find_tool(name: str) -> str:
    """Return the path of an FFmpeg tool, from PATH or Homebrew's prefix."""
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        candidate = Path(prefix) / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"{name} not found on PATH")


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata of the first video stream."""

    path: Path
    width: int
    height: int
    rotation: int
    time_base: str
    fps_nominal: float
    duration_s: float | None
    codec: str


def _fraction(text: str | None) -> Fraction | None:
    if not text:
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            return int(round(float(side["rotation"])))
    rotate = (stream.get("tags") or {}).get("rotate")
    return int(round(float(rotate))) if rotate is not None else 0


def probe_video(path: str | Path) -> VideoInfo:
    """Read the frame size, rotation, time base, and frame rate with ffprobe.

    ``fps_nominal`` is ffprobe's ``r_frame_rate``, the rate the timestamps are
    laid out for, which is the recording mode's rate on a phone; when it is
    missing, ``avg_frame_rate`` is used.

    Args:
        path: The video file.

    Returns:
        The metadata of the first video stream.

    Raises:
        ValueError: If the file has no video stream.
    """
    path = Path(path)
    cmd = [
        find_tool("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout  # noqa: S603
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"{path} has no video stream")
    s = streams[0]
    rate = _fraction(s.get("r_frame_rate")) or _fraction(s.get("avg_frame_rate"))
    tb = _fraction(s.get("time_base"))
    if tb is None:
        raise ValueError(f"{path} has no valid time base")
    duration = s.get("duration") or (data.get("format") or {}).get("duration")
    return VideoInfo(
        path=path,
        width=int(s["width"]),
        height=int(s["height"]),
        rotation=_rotation(s),
        time_base=f"{tb.numerator}/{tb.denominator}",
        fps_nominal=float(rate) if rate else float("nan"),
        duration_s=float(duration) if duration is not None else None,
        codec=str(s.get("codec_name", "")),
    )


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 checksum of a file as 64 lower-case hex digits."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def ffmpeg_version() -> str:
    """Return the first line of ``ffmpeg -version``."""
    out = subprocess.run(
        [find_tool("ffmpeg"), "-version"],
        capture_output=True,  # noqa: S603
        text=True,
        check=True,
    ).stdout
    return out.splitlines()[0].strip()


_PTS = re.compile(rb"\bn:\s*\d+\s+pts:\s*(-?\d+)\s")
_TB = re.compile(rb"config in time_base:\s*(\d+)/(\d+)")
_END = object()


def decode_frames(
    path: str | Path,
    info: VideoInfo,
    *,
    crop: tuple[int, int, int, int] | None = None,
    scale: int = 1,
    key_only: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode the video and yield ``(pts, rgb)`` per frame.

    The frame is decoded without autorotation, with the frame rate passed
    through, and piped as raw RGB. The timestamp is read from FFmpeg's
    ``showinfo`` filter and converted to the stream time base of ``info``.

    Args:
        path: The video file.
        info: Its metadata from :func:`probe_video`.
        crop: ``(width, height, x, y)`` of a region in full-frame px, taken
            before scaling. Use even ``x`` and ``y`` for chroma-subsampled
            video, where FFmpeg rounds odd offsets down.
        scale: Integer factor the frame (or crop) is shrunk by.
        key_only: Decode the key frames only.

    Yields:
        The timestamp in units of ``info.time_base`` and the (rows, columns, 3)
        uint8 frame.

    Raises:
        RuntimeError: If FFmpeg fails or stops sending timestamps.
    """
    w, h = (crop[0], crop[1]) if crop else (info.width, info.height)
    filters = []
    if crop:
        filters.append(f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}")
    if scale != 1:
        w, h = w // scale, h // scale
        filters.append(f"scale={w}:{h}:flags=area")
    filters.append("showinfo")
    cmd = [
        find_tool("ffmpeg"),
        "-hide_banner",
        "-nostats",
        "-v",
        "info",
        "-noautorotate",
    ]
    if key_only:
        cmd += ["-skip_frame", "nokey"]
    cmd += [
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
        "-vf",
        ",".join(filters),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    stream_tb = Fraction(info.time_base)
    pts_queue: queue.Queue = queue.Queue()
    errors: list[bytes] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
    assert proc.stdout is not None and proc.stderr is not None

    def read_stderr() -> None:
        filter_tb = stream_tb
        assert proc.stderr is not None
        for line in iter(proc.stderr.readline, b""):
            m = _PTS.search(line)
            if m is not None:
                pts = Fraction(int(m.group(1))) * filter_tb / stream_tb
                pts_queue.put(round(pts))
                continue
            m = _TB.search(line)
            if m is not None and int(m.group(1)) > 0:
                filter_tb = Fraction(int(m.group(1)), int(m.group(2)))
                continue
            if b"Error" in line or b"error" in line:
                errors.append(line)
        pts_queue.put(_END)

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    nbytes = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                # End of stream: let FFmpeg exit by itself so its status counts.
                proc.wait()
                break
            try:
                pts = pts_queue.get(timeout=120)
            except queue.Empty as exc:
                raise RuntimeError(
                    f"no timestamp from FFmpeg for a frame of {path}"
                ) from exc
            if pts is _END:
                raise RuntimeError(
                    f"FFmpeg sent more frames than timestamps for {path}"
                )
            yield int(pts), np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        if proc.poll() is None:
            proc.stdout.close()
            proc.kill()
        proc.wait()
        reader.join(timeout=10)
    if proc.returncode != 0:
        detail = b"".join(errors[-5:]).decode(errors="replace")
        raise RuntimeError(
            f"FFmpeg failed on {path} (exit {proc.returncode}): {detail}"
        )
