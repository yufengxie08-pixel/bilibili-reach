# -*- coding: utf-8 -*-
"""Whisper audio transcription with Groq → OpenAI fallback.

Downloads audio (yt-dlp), compresses + chunks (ffmpeg), and posts to a
Whisper-compatible API. Defaults to Groq's free `whisper-large-v3` and falls
back to OpenAI's `whisper-1` on HTTP error.

Public entry point:
    transcribe(source, *, provider="auto", out_dir=None, config=None) -> str

Designed to be importable from channels (e.g. YouTubeChannel.transcribe).
"""

from __future__ import annotations

import ipaddress
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests

from agent_reach.config import Config

# Whisper API limit is 25MB; leave headroom for multipart overhead.
SIZE_LIMIT_BYTES = 24 * 1024 * 1024
CHUNK_SECONDS = 600  # 10 min — small enough that boundary cuts rarely lose meaning

PROVIDERS = {
    "groq": {
        "endpoint": "https://api.groq.com/openai/v1/audio/transcriptions",
        "model": "whisper-large-v3",
        "key_field": "groq_api_key",
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1/audio/transcriptions",
        "model": "whisper-1",
        "key_field": "openai_api_key",
    },
}


class TranscribeError(RuntimeError):
    """Raised when transcription cannot complete."""


class MissingDependency(TranscribeError):
    """Raised when a required external binary is missing."""


class NoProviderConfigured(TranscribeError):
    """Raised when no provider has an API key configured."""


_BLOCKED_HOSTS = {
    "localhost",
    "metadata.google.internal",
}


def _require(binary: str) -> None:
    if not shutil.which(binary):
        raise MissingDependency(f"{binary} not found in PATH")


def _run(cmd: List[str], timeout: int = 600) -> None:
    """Run a subprocess, raising TranscribeError on nonzero exit or timeout.

    cmd carries user-supplied URLs/paths into yt-dlp/ffmpeg — a stalled
    network read or a hung probe must not block the CLI forever.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TranscribeError(f"{cmd[0]} timed out after {timeout}s")
    if proc.returncode != 0:
        raise TranscribeError(
            f"{cmd[0]} failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}"
        )


def _is_private_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_reserved,
            ip.is_multicast,
            ip.is_unspecified,
        )
    )


def _assert_safe_public_url(url: str) -> None:
    """Reject literal local/internal URLs without DNS-resolving public hosts."""
    if "://" not in url:
        before_slash = url.split("/", 1)[0]
        if ":" in before_slash:
            host_part, port_part = before_slash.rsplit(":", 1)
            if not host_part or not port_part.isdigit():
                raise TranscribeError("SSRF blocked: only public http(s) URLs are allowed")
        parsed = urlparse(f"https://{url}")
    else:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise TranscribeError("SSRF blocked: only public http(s) URLs are allowed")

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise TranscribeError("SSRF blocked: URL host is missing")
    if host in _BLOCKED_HOSTS or host.endswith(".localhost"):
        raise TranscribeError("SSRF blocked: internal host is not allowed")
    if _is_private_ip(host):
        raise TranscribeError("SSRF blocked: private/internal IP is not allowed")


def download_audio(url: str, out_dir: Path) -> Path:
    """Download audio with yt-dlp into out_dir; return the resulting file path."""
    _assert_safe_public_url(url)
    _require("yt-dlp")
    template = out_dir / "source.%(ext)s"
    _run(
        [
            "yt-dlp",
            "-x",
            "--audio-format",
            "m4a",
            "--audio-quality",
            "0",
            "-o",
            str(template),
            "--",
            url,
        ],
        timeout=1800,  # long podcasts over slow networks — generous but bounded
    )
    files = sorted(out_dir.glob("source.*"))
    if not files:
        raise TranscribeError("yt-dlp produced no output file")
    return files[0]


def _looks_like_bilibili_source(source: str) -> bool:
    value = source.strip()
    if value.startswith("BV") and " " not in value:
        return True
    if "b23.tv" in value:
        return True
    if "bilibili.com/video/" in value:
        return True
    return False


def download_bilibili_audio(source: str, out_dir: Path) -> List[Path]:
    """Download Bilibili audio, preferring bili-cli and falling back to public APIs."""
    try:
        return _download_bilibili_audio_via_bili_cli(source, out_dir)
    except TranscribeError:
        return _download_bilibili_audio_via_api(source, out_dir)


def _download_bilibili_audio_via_bili_cli(source: str, out_dir: Path) -> List[Path]:
    _require("bili")
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in out_dir.rglob("*") if p.is_file()}
    _run(
        [
            "bili",
            "audio",
            source,
            "-o",
            str(out_dir),
        ],
        timeout=1800,
    )
    candidates = _collect_audio_candidates(out_dir, before)
    if not candidates:
        raise TranscribeError("bili audio produced no audio file")
    return candidates


def _collect_audio_candidates(out_dir: Path, before: set[Path]) -> List[Path]:
    candidates = [
        p for p in out_dir.rglob("*")
        if p.is_file()
        and p.resolve() not in before
        and p.suffix.lower() in {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4s"}
    ]
    if not candidates:
        candidates = [
            p for p in out_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4s"}
        ]
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _download_bilibili_audio_via_api(source: str, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bvid = _extract_bilibili_bvid(source)
    cid = _get_bilibili_cid(bvid)
    audio_url = _get_bilibili_audio_url(bvid, cid)
    dst = out_dir / f"{bvid}.m4a"
    _download_url(audio_url, dst, referer=f"https://www.bilibili.com/video/{bvid}")
    return [dst]


def _extract_bilibili_bvid(source: str) -> str:
    value = source.strip()
    if value.startswith("BV") and " " not in value:
        return value
    parsed = urlparse(value)
    parts = [p for p in parsed.path.split("/") if p]
    for part in parts:
        if part.startswith("BV"):
            return part
    raise TranscribeError("could not extract Bilibili BV id from source")


def _bili_request_json(url: str, *, referer: str = "https://www.bilibili.com/") -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": referer,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise TranscribeError(f"Bilibili API request failed: {e}") from e


def _get_bilibili_cid(bvid: str) -> int:
    data = _bili_request_json(
        f"https://api.bilibili.com/x/player/pagelist?bvid={bvid}",
        referer=f"https://www.bilibili.com/video/{bvid}",
    )
    pages = data.get("data") or []
    if data.get("code") != 0 or not pages:
        raise TranscribeError(f"could not get Bilibili cid for {bvid}")
    return int(pages[0]["cid"])


def _get_bilibili_audio_url(bvid: str, cid: int) -> str:
    url = (
        "https://api.bilibili.com/x/player/playurl"
        f"?bvid={bvid}&cid={cid}&fnval=16&qn=64&otype=json&fourk=1"
    )
    data = _bili_request_json(url, referer=f"https://www.bilibili.com/video/{bvid}")
    if data.get("code") != 0:
        raise TranscribeError(f"could not get Bilibili playurl for {bvid}: {data.get('message')}")
    dash = (data.get("data") or {}).get("dash") or {}
    audios = dash.get("audio") or []
    if not audios:
        raise TranscribeError(f"no audio streams found for {bvid}")
    best = max(audios, key=lambda item: item.get("bandwidth", 0))
    base_url = best.get("baseUrl") or best.get("base_url")
    if not base_url:
        raise TranscribeError(f"audio stream missing url for {bvid}")
    return base_url


def _download_url(url: str, dst: Path, *, referer: str) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": referer,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, dst.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as e:
        raise TranscribeError(f"audio download failed: {e}") from e


def compress_audio(src: Path, out_dir: Path) -> Path:
    """Re-encode to mono / 16kHz / 32kbps m4a — keeps most content under 25MB."""
    _require("ffmpeg")
    dst = out_dir / "compressed.m4a"
    _run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            str(dst),
        ]
    )
    return dst


def chunk_audio(src: Path, out_dir: Path, segment_seconds: int = CHUNK_SECONDS) -> List[Path]:
    """Split src into segments. Re-encodes each segment so cuts align to keyframes."""
    _require("ffmpeg")
    pattern = out_dir / "chunk_%03d.m4a"
    _run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            str(pattern),
        ]
    )
    chunks = sorted(out_dir.glob("chunk_*.m4a"))
    if not chunks:
        raise TranscribeError("ffmpeg produced no chunks")
    return chunks


def _provider_key(provider: str, config: Config) -> Optional[str]:
    field = PROVIDERS[provider]["key_field"]
    val = config.get(field)
    return val or None


def transcribe_chunk(
    chunk: Path,
    provider: str,
    *,
    config: Optional[Config] = None,
    timeout: int = 120,
) -> str:
    """Transcribe one chunk via the named provider. Raises TranscribeError on failure."""
    if provider not in PROVIDERS:
        raise TranscribeError(f"unknown provider: {provider}")
    cfg = config or Config()
    key = _provider_key(provider, cfg)
    if not key:
        raise NoProviderConfigured(
            f"{provider}: missing {PROVIDERS[provider]['key_field']} "
            f"(configure with `agent-reach configure {provider}-key ...`)"
        )

    info = PROVIDERS[provider]
    with chunk.open("rb") as fh:
        try:
            resp = requests.post(
                info["endpoint"],
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (chunk.name, fh, "audio/m4a")},
                data={"model": info["model"], "response_format": "text"},
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise TranscribeError(f"{provider}: network error: {e}") from e

    if not resp.ok:
        raise TranscribeError(f"{provider}: HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.text


def _provider_order(provider: str) -> List[str]:
    if provider == "auto":
        return ["groq", "openai"]
    if provider in PROVIDERS:
        return [provider]
    raise TranscribeError(f"unknown provider: {provider} (use groq|openai|auto)")


def transcribe(
    source: str,
    *,
    provider: str = "auto",
    out_dir: Optional[Path] = None,
    config: Optional[Config] = None,
) -> str:
    """Transcribe a URL or local file path. Returns the joined transcript text.

    `provider` is one of `auto` (groq → openai), `groq`, or `openai`.
    `out_dir` defaults to a fresh temp directory; intermediate files stay there.
    """
    cfg = config or Config()
    order = _provider_order(provider)

    # Validate at least one provider is configured before doing expensive work.
    if not any(_provider_key(p, cfg) for p in order):
        names = ", ".join(PROVIDERS[p]["key_field"] for p in order)
        raise NoProviderConfigured(f"no provider key configured (need one of: {names})")

    if out_dir:
        return _transcribe_in_dir(source, order, cfg, Path(out_dir))

    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
        return _transcribe_in_dir(source, order, cfg, Path(tmp))


def _transcribe_in_dir(source: str, order: List[str], cfg: Config, work_dir: Path) -> str:
    work_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(source)
    if src_path.is_file():
        chunks = _prepare_local_audio_chunks(src_path, work_dir)
    elif _looks_like_bilibili_source(source):
        chunks = download_bilibili_audio(source, work_dir)
    else:
        audio = download_audio(source, work_dir)
        chunks = _prepare_local_audio_chunks(audio, work_dir)

    pieces: List[str] = []
    for chunk in chunks:
        text = _transcribe_with_fallback(chunk, order, cfg)
        pieces.append(text.strip())
    return "\n".join(p for p in pieces if p)


def _prepare_local_audio_chunks(audio: Path, work_dir: Path) -> List[Path]:
    """Normalize a local/remote-downloaded audio file into Whisper-sized chunks."""
    compressed = compress_audio(audio, work_dir)
    if compressed.stat().st_size <= SIZE_LIMIT_BYTES:
        return [compressed]
    return chunk_audio(compressed, work_dir)


def _transcribe_with_fallback(chunk: Path, order: List[str], config: Config) -> str:
    """Try each provider in order; return first success or raise the last error."""
    last_err: Optional[Exception] = None
    for p in order:
        if not _provider_key(p, config):
            # Skip silently — caller already validated at least one is configured.
            continue
        try:
            return transcribe_chunk(chunk, p, config=config)
        except TranscribeError as e:
            last_err = e
            continue
    raise TranscribeError(f"all providers failed for {chunk.name}: {last_err}")
