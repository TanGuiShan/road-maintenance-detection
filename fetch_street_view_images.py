"""
fetch_street_view_images.py

Batch collector for road-condition images via the Google Street View
Static API.

WHY THIS APPROACH INSTEAD OF MANUAL SCREENSHOTS
-------------------------------------------------
Manually navigating Street View and screenshotting gives you full-browser
images with UI chrome (search bar, minimap, zoom controls) that then need
cropping, and no structured location metadata. This script calls the
Static API directly: you get a clean image (no UI at all) plus exact
lat/lng/heading/date metadata for free, for every point.

HOW POINTS ARE GENERATED
-------------------------------------------------
You do not need to hand-pick every camera position. Instead, you supply
a START and END coordinate for each road segment (easiest way to get
these: right-click the road in Google Maps -> "What's here?" -> copy the
lat, lng shown). The script linearly interpolates N evenly-spaced points
between them and computes the compass heading at each point so the
camera faces along the direction of travel.

This works well for roads that are roughly straight between the two
endpoints. For a road that curves noticeably, just break it into two or
three shorter segments (each with its own start/end pair) rather than
one long one -- add multiple rows to your segments CSV.

HOW TO GET AN API KEY (required)
-------------------------------------------------
1. Go to https://console.cloud.google.com and create/select a project.
2. Enable billing on the project (required by Google, even though a free
   monthly usage threshold covers small projects like this one).
3. APIs & Services > Library > search "Street View Static API" > Enable.
4. APIs & Services > Credentials > Create Credentials > API key.
5. Restrict the key to "Street View Static API" only (Credentials page
   > click the key > API restrictions).
6. Set it as an environment variable rather than pasting it into this file:

   macOS/Linux:            export GOOGLE_STREETVIEW_KEY="your_key_here"
   Windows (PowerShell):   $env:GOOGLE_STREETVIEW_KEY="your_key_here"

INPUT FORMAT
-------------------------------------------------
Create a CSV (default name: segments.csv) with these columns:

   segment_label,start_lat,start_lng,end_lat,end_lng,points

Example:
   segment_label,start_lat,start_lng,end_lat,end_lng,points
   nanyang_ave_north,1.34845,103.68219,1.35012,103.68301,5
   jurong_west_st52,1.34921,103.70588,1.35105,103.70701,4

"points" is how many evenly-spaced Street View images to fetch along
that segment (start and end inclusive).

OUTPUT
-------------------------------------------------
Images saved under street_view_images/<segment_label>/, named
<segment_label>_<index>.jpg. A single running manifest.csv at the
output root records lat, lng, heading, capture date, and the exact
image URL requested (minus the key) for every image -- this is your
ground-truth location log, ready for you to add a good/poor judgment
column to.

USAGE
-------------------------------------------------
   python3 fetch_street_view_images.py --segments segments.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    api_key: str
    metadata_url: str = "https://maps.googleapis.com/maps/api/streetview/metadata"
    image_url: str = "https://maps.googleapis.com/maps/api/streetview"
    output_root: Path = Path("street_view_images")
    manifest_filename: str = "manifest.csv"
    image_size: str = "640x640"   # max allowed by the API
    fov: int = 90
    pitch: int = 0
    request_timeout_seconds: int = 15
    max_retries: int = 3
    seconds_between_requests: float = 0.2  # be polite to the API


def load_config() -> Config:
    api_key = os.environ.get("GOOGLE_STREETVIEW_KEY", "").strip()
    if not api_key:
        sys.exit(
            "ERROR: No Street View API key found.\n"
            "Set the GOOGLE_STREETVIEW_KEY environment variable before running.\n"
            "See the module docstring at the top of this file for setup steps."
        )
    return Config(api_key=api_key)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("streetview_collector")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# HTTP session with retry/backoff
# ---------------------------------------------------------------------------

def build_session(config: Config) -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=config.max_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# Geometry: interpolate points along a segment + compute heading
# ---------------------------------------------------------------------------

def interpolate_points(
    start_lat: float, start_lng: float,
    end_lat: float, end_lng: float,
    n_points: int,
) -> list[tuple[float, float]]:
    """Evenly-spaced lat/lng points from start to end, inclusive.
    Simple linear interpolation -- fine for short road segments where
    the curvature of the earth is negligible."""
    if n_points < 2:
        return [(start_lat, start_lng)]
    points = []
    for i in range(n_points):
        t = i / (n_points - 1)
        lat = start_lat + t * (end_lat - start_lat)
        lng = start_lng + t * (end_lng - start_lng)
        points.append((lat, lng))
    return points


def compute_heading(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compass bearing (0-360, 0=North) from point 1 to point 2, so the
    camera faces along the direction of travel."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lng = math.radians(lng2 - lng1)
    x = math.sin(delta_lng) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lng)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def check_metadata(session: requests.Session, config: Config, lat: float, lng: float, heading: float, logger: logging.Logger) -> dict:
    """Checks whether real Street View imagery exists near this point
    BEFORE spending an image request -- avoids paying for/saving blank
    'no imagery available' placeholder images."""
    params = {
        "location": f"{lat},{lng}",
        "heading": heading,
        "key": config.api_key,
    }
    response = session.get(config.metadata_url, params=params, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    return response.json()


def download_image(session: requests.Session, config: Config, lat: float, lng: float, heading: float, dest_path: Path, logger: logging.Logger) -> bool:
    params = {
        "size": config.image_size,
        "location": f"{lat},{lng}",
        "heading": heading,
        "pitch": config.pitch,
        "fov": config.fov,
        "key": config.api_key,
    }
    try:
        response = session.get(config.image_url, params=params, timeout=config.request_timeout_seconds)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return True
    except requests.RequestException as exc:
        logger.error(f"Failed to download image at {lat},{lng}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "segment_label", "point_index", "latitude", "longitude", "heading",
    "pano_id", "capture_date", "image_filename", "relative_path",
]


def append_to_manifest(config: Config, rows: list[dict]) -> None:
    manifest_path = config.output_root / config.manifest_filename
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def read_segments(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_collection(segments_path: Path) -> None:
    config = load_config()
    logger = setup_logging()
    session = build_session(config)

    segments = read_segments(segments_path)
    if not segments:
        logger.error(f"No segments found in {segments_path}")
        return

    manifest_rows = []
    total_saved = 0
    total_skipped_no_imagery = 0

    for seg in segments:
        label = seg["segment_label"].strip()
        start_lat, start_lng = float(seg["start_lat"]), float(seg["start_lng"])
        end_lat, end_lng = float(seg["end_lat"]), float(seg["end_lng"])
        n_points = int(seg["points"])

        points = interpolate_points(start_lat, start_lng, end_lat, end_lng, n_points)
        seg_dir = config.output_root / label
        seg_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Segment '{label}': {len(points)} points")

        for idx, (lat, lng) in enumerate(points, start=1):
            heading = compute_heading(start_lat, start_lng, end_lat, end_lng)

            meta = check_metadata(session, config, lat, lng, heading, logger)
            if meta.get("status") != "OK":
                logger.warning(f"  [{idx}] No imagery at {lat:.5f},{lng:.5f} (status={meta.get('status')}) -- skipping")
                total_skipped_no_imagery += 1
                time.sleep(config.seconds_between_requests)
                continue

            image_filename = f"{label}_{idx}.jpg"
            dest_path = seg_dir / image_filename

            if download_image(session, config, lat, lng, heading, dest_path, logger):
                total_saved += 1
                manifest_rows.append({
                    "segment_label": label,
                    "point_index": idx,
                    "latitude": lat,
                    "longitude": lng,
                    "heading": round(heading, 1),
                    "pano_id": meta.get("pano_id", ""),
                    "capture_date": meta.get("date", ""),
                    "image_filename": image_filename,
                    "relative_path": str(dest_path.relative_to(config.output_root)),
                })
                logger.info(f"  [{idx}] Saved {image_filename} (captured {meta.get('date', 'unknown')})")

            time.sleep(config.seconds_between_requests)

    append_to_manifest(config, manifest_rows)
    logger.info(
        f"Done. {total_saved} images saved, {total_skipped_no_imagery} points skipped "
        f"(no imagery). Manifest at '{config.output_root / config.manifest_filename}'."
    )
    logger.info(
        "Next step: open the manifest CSV and add a 'condition_label' column "
        "(e.g. good/poor) as you review each image -- this becomes your ground truth."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-fetch Street View images along road segments.")
    parser.add_argument("--segments", type=Path, default=Path("segments.csv"), help="Path to segments CSV.")
    args = parser.parse_args()
    run_collection(args.segments)