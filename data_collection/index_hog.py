import os
import sqlite3
import numpy as np
from PIL import Image
from skimage.feature import hog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.auto import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZE = 64
VECTOR_SIZE = 1568  # 7x7 blocks * 2x2 cells * 8 orientations
QDRANT_COLLECTION = "satellite_patches_hog"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "metadata.sqlite")
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_hog")

# ── Init ─────────────────────────────────────────────────────────────────────
# Qdrant Client
os.makedirs(QDRANT_PATH, exist_ok=True)
qdrant = QdrantClient(path=QDRANT_PATH)

existing = {c.name: c for c in qdrant.get_collections().collections}
if QDRANT_COLLECTION in existing:
    info = qdrant.get_collection(QDRANT_COLLECTION)
    current_size = info.config.params.vectors.size
    if current_size != VECTOR_SIZE:
        print(f"⚠️  Vector size changed ({current_size} → {VECTOR_SIZE}). Recreating collection...")
        qdrant.delete_collection(QDRANT_COLLECTION)
        del existing[QDRANT_COLLECTION]
        # Reset indexed_hog flags so all images get re-indexed
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE metadata SET indexed_hog = 0")
        conn.commit()
        conn.close()
        print("   Reset all indexed_hog flags.")

if QDRANT_COLLECTION not in existing:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ Created fresh Qdrant collection '{QDRANT_COLLECTION}'.")

# ── Process ──────────────────────────────────────────────────────────────────
def get_unindexed_images():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Ensure table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    if not c.fetchone():
        conn.close()
        return []
    
    c.execute("SELECT id, file_path, lat, lon, date FROM metadata WHERE indexed_hog = 0")
    rows = c.fetchall()
    conn.close()
    return rows

def mark_as_indexed(ids):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany("UPDATE metadata SET indexed_hog = 1 WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()

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
    
    # L2 normalize just to be consistent with cosine distance usage
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
        
    return features.astype(np.float32)

def main():
    rows = get_unindexed_images()
    if not rows:
        print("✅ No unindexed images found. Everything is up to date!")
        return

    print(f"Found {len(rows)} images to index with HOG.")
    
    base_dir = os.path.dirname(__file__)
    
    for i in tqdm(range(0, len(rows), BATCH_SIZE)):
        batch = rows[i:i + BATCH_SIZE]
        
        points = []
        ids_to_mark = []
        
        for row in batch:
            id_, rel_path, lat, lon, date_str = row
            abs_path = os.path.join(base_dir, rel_path)
            try:
                img = Image.open(abs_path)
                features = compute_hog_feature(img)
                
                points.append(
                    PointStruct(
                        id=id_,
                        vector=features.tolist(),
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
