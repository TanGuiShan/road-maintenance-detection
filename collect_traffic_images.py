"""
collect_traffic_images.py

Weekly collector for LTA Traffic Camera images via LTA DataMall's real-time API.

WHY DATAMALL AND NOT data.gov.sg
------------------------------------
An earlier version of this script used data.gov.sg's proxy endpoint
(api-open.data.gov.sg/v2/real-time/api/traffic-images). That endpoint
returned 403 Forbidden from GitHub Actions runners regardless of API key
or request headers used — consistent with cloud/datacenter IP ranges
being blocked by a WAF or bot-detection layer in front of it. LTA
DataMall (datamall2.mytransport.sg) is the original authoritative source
for this same data, on a different domain with different infrastructure,
so it's the next thing worth trying. THIS IS NOT GUARANTEED TO WORK — if
you still get blocked here, the problem is very likely GitHub Actions'
IP ranges specifically, and the real fix is running the collector
somewhere with a non-datacenter IP (see the scheduling section below).

WHAT THIS SCRIPT DOES
----------------------
1. Calls the LTA DataMall real-time Traffic Images API.
2. Downloads every camera's current image.
3. Saves each image under a dated folder, plus a CSV manifest recording
   camera_id, road location, timestamp, and file path for every image
   (so you have a queryable index for your FYP dataset, not just a folder
   of unlabelled JPEGs).

HOW TO GET AN ACCOUNT KEY (required)
----------------------------------------
Unlike data.gov.sg, DataMall requires a key for all requests, not just
for higher rate limits.

1. Go to https://datamall.lta.gov.sg/content/datamall/en.html
2. Look for "Request for API Access" / "Register" (free, instant access
   for most datasets — LTA emails you an Account Key).
3. Set it as an environment variable rather than pasting it into this file
   (keeps it out of version control if you push this to GitHub):

   macOS/Linux:   export LTA_ACCOUNT_KEY="your_key_here"
   Windows (PowerShell): $env:LTA_ACCOUNT_KEY="your_key_here"

This gets sent as the 'AccountKey' header on every request (DataMall's
own auth scheme — different from data.gov.sg's 'x-api-key').

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
    account_key: str
    api_url: str = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    output_root: Path = Path("traffic_images")
    manifest_filename: str = "manifest.csv"
    request_timeout_seconds: int = 15
    max_retries: int = 3


def load_config() -> Config:
    """Reads the DataMall Account Key from the environment. Fails fast
    with a clear message if it's missing — unlike data.gov.sg, DataMall
    requires a key on every request, so there's no keyless fallback here.
    """
    account_key = os.environ.get("LTA_ACCOUNT_KEY", "").strip()

    # Uncomment the line below ONLY for quick local testing.
    # Do not commit a real key to source control.
    # account_key = account_key or "PASTE_YOUR_KEY_HERE_FOR_TESTING_ONLY"

    if not account_key:
        sys.exit(
            "ERROR: No DataMall Account Key found.\n"
            "Set the LTA_ACCOUNT_KEY environment variable before running.\n"
            "See the module docstring at the top of this file for setup steps."
        )
    return Config(account_key=account_key)


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
    session.headers.update({
        "AccountKey": config.account_key,
        "accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
    })
    return session


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def fetch_camera_snapshot(session: requests.Session, config: Config, logger: logging.Logger) -> list[dict]:
    """Calls the DataMall Traffic-Imagesv2 API and returns the list of
    camera records for the current snapshot.

    IMPORTANT — image link expiry: DataMall's docs state ImageLink URLs
    are only valid for a few minutes after this call. Download images
    promptly after fetching the snapshot (run_collection does this —
    don't insert a long delay between fetch and download if you modify
    this script).
    """
    logger.info("Requesting current traffic camera snapshot from LTA DataMall...")
    response = session.get(config.api_url, timeout=config.request_timeout_seconds)

    if response.status_code == 403:
        # Unlike the earlier data.gov.sg version, a 403 here is NOT
        # something we can retry around — DataMall requires a valid key
        # on every call, so this means either the key is wrong/inactive,
        # or (less likely, since this is a different domain/infra to
        # data.gov.sg) this network is also being blocked. Surface both
        # possibilities rather than guessing.
        raise RuntimeError(
            "403 Forbidden from DataMall. Most likely causes: (1) LTA_ACCOUNT_KEY "
            "is missing/wrong/not yet activated — new keys can take a short "
            "while to activate after signup, or (2) this network's IP range "
            "is blocked, same class of issue as the earlier data.gov.sg 403s. "
            "Check the key first; if a known-good key still gets 403 here, "
            "it points to (2)."
        )

    response.raise_for_status()
    payload = response.json()

    # NOTE: DataMall wraps results in a flat "value" list — a simpler shape
    # than data.gov.sg's nested data->items->cameras. If LTA changes this
    # schema, print(payload) once to inspect the actual structure.
    cameras = payload.get("value", [])
    if not cameras:
        logger.warning("API responded successfully but returned no cameras.")
        return []

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
        camera_id = camera.get("CameraID", "unknown")
        image_url = camera.get("ImageLink")
        camera_timestamp = camera.get("Timestamp", "")

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
                "latitude": camera.get("Latitude", ""),
                "longitude": camera.get("Longitude", ""),
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