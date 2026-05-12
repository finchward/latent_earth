import os
import sqlite3
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2lab
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.auto import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZE = 64
HOG_VECTOR_SIZE = 1568   # 7x7 blocks * 2x2 cells * 8 orientations
COLOUR_VECTOR_SIZE = 192   # 8x8 grid * 3 channels (R, G, B)
QDRANT_COLLECTION = "satellite_patches_hog_hybrid"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "metadata.sqlite")
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_hog_hybrid")

# ── Init ─────────────────────────────────────────────────────────────────────
# Ensure indexed_hog_hybrid column exists
conn = sqlite3.connect(DB_PATH)
try:
    conn.execute("ALTER TABLE metadata ADD COLUMN indexed_hog_hybrid INTEGER DEFAULT 0")
    conn.commit()
    print("✅ Added 'indexed_hog_hybrid' column to metadata table.")
except sqlite3.OperationalError:
    pass  # column already exists
conn.close()

# Qdrant Client
os.makedirs(QDRANT_PATH, exist_ok=True)
qdrant = QdrantClient(path=QDRANT_PATH)

existing = {c.name: c for c in qdrant.get_collections().collections}
if QDRANT_COLLECTION in existing:
    info = qdrant.get_collection(QDRANT_COLLECTION)
    # Named vectors — config is a dict
    vec_cfg = info.config.params.vectors
    needs_recreate = False
    if isinstance(vec_cfg, dict):
        hog_cfg = vec_cfg.get("hog")
        colour_cfg = vec_cfg.get("colour")
        if hog_cfg is None or colour_cfg is None:
            needs_recreate = True
        elif hog_cfg.size != HOG_VECTOR_SIZE or colour_cfg.size != COLOUR_VECTOR_SIZE:
            needs_recreate = True
        elif not hog_cfg.on_disk or not colour_cfg.on_disk:
            needs_recreate = True
    else:
        # Old single-vector collection, recreate
        needs_recreate = True

    if needs_recreate:
        print(f"⚠️  Vector config changed. Recreating collection...")
        
        # In Qdrant Local, delete_collection can be unreliable on Windows due to file locks.
        # We explicitly close the client, wipe the folder, and start fresh.
        qdrant.close()
        import shutil
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)
        os.makedirs(QDRANT_PATH, exist_ok=True)
        
        qdrant = QdrantClient(path=QDRANT_PATH)
        existing = {}
        
        # Reset indexed flags so all images get re-indexed
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE metadata SET indexed_hog_hybrid = 0")
        conn.commit()
        conn.close()
        print("   Reset all indexed_hog_hybrid flags.")

if QDRANT_COLLECTION not in existing:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "hog": VectorParams(size=HOG_VECTOR_SIZE, distance=Distance.EUCLID, on_disk=True),
            "colour": VectorParams(size=COLOUR_VECTOR_SIZE, distance=Distance.EUCLID, on_disk=True),
        },
    )
    print(f"✅ Created fresh Qdrant collection '{QDRANT_COLLECTION}' (On-Disk) with named vectors [hog, colour].")

# ── Feature Computation ──────────────────────────────────────────────────────
def compute_hog_feature(img: Image.Image) -> np.ndarray:
    """
    Computes a HOG feature vector for an image.
    Resizes to 128x128.
    Using 8 orientations, 16x16 pixels per cell, 2x2 cells per block.
    Block normalization (L2-Hys) makes the descriptor invariant to
    lighting differences — critical for matching face edges to landscape edges.
    Result: 7x7 blocks * 2x2 cells * 8 orientations = 1568 features.
    """
    img_gray = img.convert("L")
    img_resized = img_gray.resize((128, 128))
    img_arr = np.array(img_resized)
    
    features = hog(
        img_arr, 
        orientations=8, 
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    # Compute overall standard deviation to capture true global edge strength/contrast
    grad_mag = np.std(img_arr.astype(np.float32))
    
    # Scale HOG vector by the gradient magnitude
    features = features * grad_mag
        
    return features.astype(np.float32)

def compute_colour_feature(img: Image.Image) -> np.ndarray:
    """
    Computes a spatial colour descriptor in LAB space by resizing to 8x8 and flattening.
    
    Using LAB space ensures that Euclidean distance corresponds to perceptual 
    colour difference (Delta E).
    """
    img_rgb = img.convert("RGB")
    img_small = img_rgb.resize((8, 8), Image.LANCZOS)
    arr_rgb = np.array(img_small, dtype=np.float32) / 255.0
    
    # Convert to LAB space
    arr_lab = rgb2lab(arr_rgb)
    
    # Flatten to (192,) vector
    features = arr_lab.flatten().astype(np.float32)
    return features

# ── Database Helpers ─────────────────────────────────────────────────────────
def get_unindexed_images():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Ensure table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    if not c.fetchone():
        conn.close()
        return []
    
    c.execute("SELECT id, file_path, lat, lon, date FROM metadata WHERE indexed_hog_hybrid = 0")
    rows = c.fetchall()
    conn.close()
    return rows

def mark_as_indexed(ids):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany("UPDATE metadata SET indexed_hog_hybrid = 1 WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    rows = get_unindexed_images()
    if not rows:
        print("✅ No unindexed images found. Everything is up to date!")
        return

    print(f"Found {len(rows)} images to index with HOG+Colour hybrid.")
    
    base_dir = os.path.dirname(__file__)
    
    for i in tqdm(range(0, len(rows), BATCH_SIZE)):
        batch = rows[i:i + BATCH_SIZE]
        
        points = []
        ids_to_mark = []
        
        for row in batch:
            id_, rel_path, lat, lon, date_str = row
            abs_path = os.path.join(base_dir, rel_path)
            try:
                img = Image.open(abs_path).convert("RGB")
                hog_vec = compute_hog_feature(img)
                colour_vec = compute_colour_feature(img)
                
                points.append(
                    PointStruct(
                        id=id_,
                        vector={
                            "hog": hog_vec.tolist(),
                            "colour": colour_vec.tolist(),
                        },
                        payload={
                            "filename": os.path.basename(rel_path),
                            "lat": lat,
                            "lon": lon,
                            "date": date_str
                        }
                    )
                )
                ids_to_mark.append(id_)
            except Exception as e:
                print(f"Failed to process image {abs_path}: {e}")
        
        if points:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
            mark_as_indexed(ids_to_mark)

    print("✅ Indexing complete.")
    qdrant.close()

if __name__ == "__main__":
    main()
