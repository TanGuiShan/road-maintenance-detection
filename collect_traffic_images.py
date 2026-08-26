"""
collect_traffic_images.py

Weekly collector for LTA Traffic Camera images via data.gov.sg's real-time API.

WHAT THIS SCRIPT DOES
----------------------
1. Calls the data.gov.sg real-time Traffic Images API.
2. Downloads every camera's current image.
3. Saves each image under a dated folder, plus a CSV manifest recording
   camera_id, road location, timestamp, and file path for every image
   (so you have a queryable index for your FYP dataset, not just a folder
   of unlabelled JPEGs).

HOW TO GET AN API KEY (one-time setup)
----------------------------------------
1. Go to https://data.gov.sg/ and click "Log in" (top right) to create a
   free account.
2. Once logged in, go to your account/API settings page and generate an
   API key. This gets sent as the 'x-api-key' header on every request.
3. Set it as an environment variable rather than pasting it into this file
   (keeps it out of version control if you push this to GitHub):

   macOS/Linux:   export DATAGOVSG_API_KEY="your_key_here"
   Windows (PowerShell): $env:DATAGOVSG_API_KEY="your_key_here"

   Or, for a quick local test only, set DEFAULT_API_KEY below.

HOW TO SCHEDULE THIS TO RUN WEEKLY
------------------------------------
This script is designed to run ONCE per invocation (fetch, download, exit).
Use your OS's scheduler to trigger it weekly — no need to keep Python running.

  Windows (Task Scheduler):
    1. Open Task Scheduler -> Create Basic Task.
    2. Trigger: Weekly, pick day/time (e.g. Monday 9:00 AM).
    3. Action: Start a program.
       Program: path to python.exe (e.g. C:\\Python311\\python.exe)
       Arguments: "C:\\path\\to\\collect_traffic_images.py"

  macOS/Linux (cron):
    1. Run: crontab -e
    2. Add a line (runs every Monday at 09:00):
       0 9 * * 1 /usr/bin/python3 /path/to/collect_traffic_images.py >> /path/to/logs/cron.log 2>&1

  GitHub Actions (recommended if your laptop won't reliably be on):
    See .github/workflows/weekly-collect.yml alongside this script. GitHub's
    servers run the job on schedule regardless of your laptop's state, and
    the API key is stored as an encrypted repo secret rather than in this
    file. The output_root path below is relative, so it works unchanged
    whether run locally or inside the Actions runner.

Re-running this script simply adds a new dated folder + new manifest rows;
nothing gets overwritten, so it's safe to schedule and forget.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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
    api_url: str = "https://api-open.data.gov.sg/v2/real-time/api/traffic-images"
    output_root: Path = Path("traffic_images")
    manifest_filename: str = "manifest.csv"
    request_timeout_seconds: int = 15
    max_retries: int = 3


def load_config() -> Config:
    """Reads the API key from the environment. Fails fast with a clear
    message if it's missing, instead of failing deep inside a request."""
    api_key = os.environ.get("DATAGOVSG_API_KEY", "").strip()

    # Uncomment the line below ONLY for quick local testing.
    # Do not commit a real key to source control.
    api_key = api_key or "PASTE_YOUR_KEY_HERE_FOR_TESTING_ONLY"

    if not api_key:
        sys.exit(
            "ERROR: No API key found.\n"
            "Set the DATAGOVSG_API_KEY environment variable before running.\n"
            "See the module docstring at the top of this file for setup steps."
        )
    return Config(api_key=api_key)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("traffic_collector")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# HTTP session with retry/backoff (traffic cams are real-time; a single
# dropped request shouldn't kill a whole weekly run)
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
    session.headers.update({"x-api-key": config.api_key})
    return session


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def fetch_camera_snapshot(session: requests.Session, config: Config, logger: logging.Logger) -> list[dict]:
    """Calls the real-time traffic-images API and returns the list of
    camera records for the current snapshot."""
    logger.info("Requesting current traffic camera snapshot...")
    response = session.get(config.api_url, timeout=config.request_timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    if payload.get("code") != 0:
        raise RuntimeError(f"API returned an error: {payload.get('errorMsg')}")

    # NOTE: data.gov.sg's v2 real-time APIs nest results under data -> items.
    # If the schema has changed since this script was written, print(payload)
    # once to inspect the actual structure and adjust the two lines below.
    items = payload.get("data", {}).get("items", [])
    if not items:
        logger.warning("API responded successfully but returned no items.")
        return []

    cameras = items[0].get("cameras", [])
    logger.info(f"Snapshot returned {len(cameras)} cameras.")
    return cameras


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def download_image(session: requests.Session, image_url: str, dest_path: Path, config: Config) -> bool:
    """Downloads a single image to dest_path. Returns True on success."""
    try:
        response = session.get(image_url, timeout=config.request_timeout_seconds)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return True
    except requests.RequestException as exc:
        logging.getLogger("traffic_collector").error(f"Failed to download {image_url}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Manifest (CSV index of every image ever collected)
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "run_timestamp",
    "camera_id",
    "latitude",
    "longitude",
    "camera_timestamp",
    "image_filename",
    "relative_path",
]


def append_to_manifest(config: Config, rows: list[dict]) -> None:
    """Appends rows to a single master manifest.csv at the output root,
    creating it with a header if it doesn't exist yet. Keeping one running
    manifest (rather than a per-week CSV) makes later analysis easier."""
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

def run_collection() -> None:
    config = load_config()
    logger = setup_logging()
    session = build_session(config)

    run_start = datetime.now()
    week_folder = config.output_root / run_start.strftime("%Y-%m-%d")
    week_folder.mkdir(parents=True, exist_ok=True)

    try:
        cameras = fetch_camera_snapshot(session, config, logger)
    except (requests.RequestException, RuntimeError) as exc:
        logger.error(f"Aborting run — could not fetch camera snapshot: {exc}")
        sys.exit(1)

    if not cameras:
        logger.warning("No cameras to process. Exiting without writing manifest rows.")
        return

    manifest_rows = []
    success_count = 0

    for camera in cameras:
        camera_id = camera.get("camera_id", "unknown")
        image_url = camera.get("image")
        location = camera.get("location", {})
        camera_timestamp = camera.get("timestamp", "")

        if not image_url:
            logger.warning(f"Camera {camera_id} has no image URL — skipping.")
            continue

        image_filename = f"{camera_id}.jpg"
        dest_path = week_folder / image_filename

        if download_image(session, image_url, dest_path, config):
            success_count += 1
            manifest_rows.append({
                "run_timestamp": run_start.isoformat(timespec="seconds"),
                "camera_id": camera_id,
                "latitude": location.get("latitude", ""),
                "longitude": location.get("longitude", ""),
                "camera_timestamp": camera_timestamp,
                "image_filename": image_filename,
                "relative_path": str(dest_path.relative_to(config.output_root)),
            })

    append_to_manifest(config, manifest_rows)

    logger.info(
        f"Done. {success_count}/{len(cameras)} images saved to '{week_folder}'. "
        f"Manifest updated at '{config.output_root / config.manifest_filename}'."
    )


if __name__ == "__main__":
    run_collection()
