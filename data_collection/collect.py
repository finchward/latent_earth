import os
import io
import math
import random
import datetime
import threading
import sqlite3
import uuid
import requests
import numpy as np
from PIL import Image
import ee
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ID = 'mercari-agent'
PATCH_SIZE_M = 5000
IMAGE_SIZE_PX = 500
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BANDS = ["B4", "B3", "B2"]          # RGB
MAX_CLOUD_PCT = 20
MIN_BRIGHTNESS = 30.0

SENTINEL_START = "2017-03-28"
WINDOW_DAYS = 30
COLLECTION_WORKERS = 100

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMG_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "metadata.sqlite")

os.makedirs(IMG_DIR, exist_ok=True)

# ── Initialization ───────────────────────────────────────────────────────────
try:
    ee.Initialize(project=PROJECT_ID)
    print(f"✅ GEE initialised: {PROJECT_ID}")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print(f"✅ GEE authenticated and initialised: {PROJECT_ID}")

SENTINEL_START_DATE = datetime.date.fromisoformat(SENTINEL_START)
TODAY = datetime.date.today()
TOTAL_DAYS = (TODAY - SENTINEL_START_DATE).days

# ── Database Setup ───────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            lat REAL,
            lon REAL,
            date TEXT,
            indexed_dinov2 BOOLEAN DEFAULT 0,
            indexed_hog BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()
    return conn

# ── Helpers ──────────────────────────────────────────────────────────────────
_LAND_IMAGE = (
    ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    .select("occurrence")
    .unmask(0)
    .lt(50)
)

def is_land(lat: float, lon: float) -> bool:
    try:
        point = ee.Geometry.Point([lon, lat])
        val = _LAND_IMAGE.sample(region=point, scale=500).first().getInfo()
        return val is not None and val["properties"].get("occurrence", 0) == 1
    except Exception as e:
        print(f"  ❌ [GEE Error in is_land] {e}")
        return False

def random_land_coord() -> tuple[float, float]:
    sin_min = math.sin(math.radians(-60))
    sin_max = math.sin(math.radians(75))
    while True:
        lon = random.uniform(-180, 180)
        lat = math.degrees(math.asin(random.uniform(sin_min, sin_max)))
        if is_land(lat, lon):
            return lat, lon

def random_date_window() -> tuple[str, str]:
    max_offset = TOTAL_DAYS - WINDOW_DAYS
    offset = random.randint(0, max_offset)
    start = SENTINEL_START_DATE + datetime.timedelta(days=offset)
    end = start + datetime.timedelta(days=WINDOW_DAYS)
    return str(start), str(end)

def is_bright_enough(img: Image.Image) -> bool:
    return np.array(img, dtype=np.float32).mean() >= MIN_BRIGHTNESS

def fetch_patch(lat: float, lon: float) -> tuple[Image.Image, str] | None:
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(PATCH_SIZE_M / 2).bounds()

        window_start, window_end = random_date_window()

        col = (
            ee.ImageCollection(COLLECTION)
            .filterBounds(region)
            .filterDate(window_start, window_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
            .select(BANDS)
        )
        if col.size().getInfo() == 0:
            return None

        scenes = col.toList(col.size())
        n_scenes = scenes.size().getInfo()
        scene = ee.Image(scenes.get(random.randint(0, n_scenes - 1)))

        ts_ms = scene.date().millis().getInfo()
        scene_date = str(datetime.date.fromtimestamp(ts_ms / 1000))

        img_ee = scene.divide(10000).multiply(255).toByte()
        url = img_ee.getThumbURL({
            "region": region,
            "dimensions": f"{IMAGE_SIZE_PX}x{IMAGE_SIZE_PX}",
            "format": "png",
            "bands": BANDS,
        })
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        return (img, scene_date) if is_bright_enough(img) else None
    except Exception as e:
        print(f"  ❌ [GEE Error in fetch_patch] {e}")
        return None

# ── Main Collection Loop ─────────────────────────────────────────────────────
def main():
    conn = init_db()
    stop_flag = False
    collected_this_session = 0
    _session_lock = threading.Lock()

    def _fetch_and_store(_):
        while not stop_flag:
            lat, lon = random_land_coord()
            result = fetch_patch(lat, lon)
            if result is None:
                continue

            img, scene_date = result
            
            # Save image
            img_filename = f"{uuid.uuid4().hex}.png"
            img_path = os.path.join(IMG_DIR, img_filename)
            img.save(img_path)
            
            # Store in db using relative path so it's portable
            rel_path = os.path.join("data", "images", img_filename)

            # Use an isolated connection for the thread, or thread-safe inserts
            try:
                # To be completely safe with sqlite in multiple threads, we create a new connection per thread write
                # But it's easier to just use a lock
                with _session_lock:
                    local_conn = sqlite3.connect(DB_PATH)
                    c = local_conn.cursor()
                    c.execute(
                        "INSERT INTO metadata (file_path, lat, lon, date) VALUES (?, ?, ?, ?)",
                        (rel_path, lat, lon, scene_date)
                    )
                    local_conn.commit()
                    local_conn.close()
                    
                    nonlocal collected_this_session
                    collected_this_session += 1
                    current_count = collected_this_session
                
                return lat, lon, scene_date, current_count
            except Exception as e:
                print(f"Error saving to db: {e}")
                if os.path.exists(img_path):
                    os.remove(img_path)

        return None

    print(f"🚀 Starting continuous collection with {COLLECTION_WORKERS} workers.")
    print(f"   Images will be saved to {IMG_DIR}")
    print(f"   Metadata will be saved to {DB_PATH}")
    print(f"   Press Ctrl+C to stop.\n")

    try:
        with ThreadPoolExecutor(max_workers=COLLECTION_WORKERS) as pool:
            while True:
                futures = [pool.submit(_fetch_and_store, i) for i in range(COLLECTION_WORKERS)]

                for f in as_completed(futures):
                    result = f.result()
                    if result is not None:
                        lat, lon, scene_date, count = result
                        print(f"  ✅ [Session: {count:>4}] ({lat:.4f}, {lon:.4f}) {scene_date}")
    except KeyboardInterrupt:
        stop_flag = True
        print(f"\n⏹  Stopped. Collected {collected_this_session} patches this session.")
        print("💾 Done.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
