"""Identification and metadata extraction for multimedia files."""

from __future__ import annotations

import mimetypes
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}


@dataclass
class MediaFileProfile:
    path: str
    media_type: str
    mime_type: str | None
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_dimensions: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultimediaProfile:
    files: List[MediaFileProfile]
    file_type_counts: Dict[str, int]
    total_size_bytes: int
    feature_dimensions: int | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": [item.to_dict() for item in self.files],
            "file_type_counts": self.file_type_counts,
            "total_size_bytes": self.total_size_bytes,
            "feature_dimensions": self.feature_dimensions,
        }


def _image_metadata(path: Path) -> Dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {"metadata_status": "Install Pillow to extract image dimensions."}

    try:
        with Image.open(path) as image:
            width, height = image.size
            return {
                "width": width,
                "height": height,
                "mode": image.mode,
                "format": image.format,
                "feature_dimensions": width * height * len(image.getbands()),
                "metadata": {
                    key: value
                    for key, value in image.info.items()
                    if isinstance(value, (str, int, float, bool))
                },
            }
    except (OSError, ValueError) as error:
        return {"metadata_error": str(error)}


def _audio_video_metadata(path: Path) -> Dict[str, Any]:
    """Extract container and stream details when ffprobe is available."""
    executable = shutil.which("ffprobe")
    if executable is None:
        return {"metadata_status": "Install FFmpeg to extract audio/video stream metadata."}
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                "format=duration,bit_rate:stream=codec_type,codec_name,width,height,channels,sample_rate,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return {"metadata_error": result.stderr.strip() or "ffprobe could not read this media file."}
        return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        return {"metadata_error": str(error)}


def inspect_media_file(path: str | Path) -> MediaFileProfile:
    """Identify one image, audio, or video file and extract available metadata."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Media file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        media_type = "image"
    elif suffix in AUDIO_EXTENSIONS:
        media_type = "audio"
    elif suffix in VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        media_type = "unknown"

    metadata: Dict[str, Any] = {"name": file_path.name, "suffix": suffix}
    dimensions = None
    if media_type == "image":
        metadata.update(_image_metadata(file_path))
        dimensions = metadata.get("feature_dimensions")
    elif media_type in {"audio", "video"}:
        metadata.update(_audio_video_metadata(file_path))
    return MediaFileProfile(
        path=str(file_path),
        media_type=media_type,
        mime_type=mimetypes.guess_type(file_path.name)[0],
        size_bytes=file_path.stat().st_size,
        metadata=metadata,
        feature_dimensions=dimensions,
    )


def inspect_media_path(path: str | Path) -> MultimediaProfile:
    """Inspect one media file or all supported media files in a directory."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    paths = [source] if source.is_file() else sorted(
        item for item in source.rglob("*") if item.is_file()
    )
    profiles = [inspect_media_file(item) for item in paths]
    profiles = [profile for profile in profiles if profile.media_type != "unknown"]
    counts: Dict[str, int] = {}
    for profile in profiles:
        counts[profile.media_type] = counts.get(profile.media_type, 0) + 1
    dimensions = [profile.feature_dimensions for profile in profiles if profile.feature_dimensions is not None]
    return MultimediaProfile(
        files=profiles,
        file_type_counts=counts,
        total_size_bytes=sum(profile.size_bytes for profile in profiles),
        feature_dimensions=max(dimensions) if dimensions else None,
    )


def analyze_multimedia_references(series: pd.Series) -> MultimediaProfile:
    """Identify media file references embedded in a tabular column."""
    profiles: List[MediaFileProfile] = []
    seen: set[str] = set()
    for raw_value in series.dropna():
        value = str(raw_value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        parsed = urlparse(value)
        filename = Path(parsed.path or value).name
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
            continue
        if Path(value).is_file():
            profiles.append(inspect_media_file(value))
            continue
        if suffix in IMAGE_EXTENSIONS:
            media_type = "image"
        elif suffix in AUDIO_EXTENSIONS:
            media_type = "audio"
        else:
            media_type = "video"
        profiles.append(
            MediaFileProfile(
                path=value,
                media_type=media_type,
                mime_type=mimetypes.guess_type(filename)[0],
                size_bytes=0,
                metadata={"name": filename, "suffix": suffix, "source": "tabular reference"},
            )
        )

    counts: Dict[str, int] = {}
    for profile in profiles:
        counts[profile.media_type] = counts.get(profile.media_type, 0) + 1
    dimensions = [profile.feature_dimensions for profile in profiles if profile.feature_dimensions is not None]
    return MultimediaProfile(
        files=profiles,
        file_type_counts=counts,
        total_size_bytes=sum(profile.size_bytes for profile in profiles),
        feature_dimensions=max(dimensions) if dimensions else None,
    )
