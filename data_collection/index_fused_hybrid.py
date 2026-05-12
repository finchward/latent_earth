import os
import sqlite3
import shutil

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2lab
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.auto import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

BATCH_SIZE         = 128
HOG_VECTOR_SIZE    = 1568   # 7x7 blocks * 2x2 cells * 8 orientations
COLOUR_VECTOR_SIZE = 192    # 8x8 grid * 3 LAB channels
FUSION_HOG_WEIGHT    = 0.577
FUSION_COLOUR_WEIGHT = 0.816   # sqrt(0.75) for 1:3 squared influence ratio

# PCA — set PCA_ENABLED = True and choose a target dimension to enable
PCA_ENABLED = True
PCA_DIM     = 256   # only used when PCA_ENABLED is True

RAW_VECTOR_SIZE = HOG_VECTOR_SIZE + COLOUR_VECTOR_SIZE   # 1760
VECTOR_SIZE     = PCA_DIM if PCA_ENABLED else RAW_VECTOR_SIZE

QDRANT_COLLECTION = "satellite_patches_fused_hybrid"
DB_COLUMN         = "indexed_fused_hybrid"

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
DB_PATH     = os.path.join(DATA_DIR, "metadata.sqlite")
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_fused_hybrid")
PCA_PATH    = os.path.join(DATA_DIR, "pca_fused_hybrid.joblib")

# ── Init — DB column ──────────────────────────────────────────────────────────

conn = sqlite3.connect(DB_PATH)
try:
    conn.execute(f"ALTER TABLE metadata ADD COLUMN {DB_COLUMN} INTEGER DEFAULT 0")
    conn.commit()
    print(f"✅ Added '{DB_COLUMN}' column to metadata table.")
except sqlite3.OperationalError:
    pass  # column already exists
conn.close()

# ── Init — Qdrant ─────────────────────────────────────────────────────────────

os.makedirs(QDRANT_PATH, exist_ok=True)
qdrant = QdrantClient(path=QDRANT_PATH)

existing = {c.name: c for c in qdrant.get_collections().collections}
if QDRANT_COLLECTION in existing:
    info    = qdrant.get_collection(QDRANT_COLLECTION)
    vec_cfg = info.config.params.vectors

    # Recreate if schema is wrong (e.g. named vectors from a previous attempt,
    # or a different vector size due to a PCA_DIM change).
    needs_recreate = False
    if isinstance(vec_cfg, dict):
        needs_recreate = True          # named-vector collection — wrong schema
    elif vec_cfg.size != VECTOR_SIZE or vec_cfg.distance != Distance.EUCLID:
        needs_recreate = True

    if needs_recreate:
        print("⚠️  Vector config mismatch. Recreating collection...")
        qdrant.close()
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)
        os.makedirs(QDRANT_PATH, exist_ok=True)
        qdrant   = QdrantClient(path=QDRANT_PATH)
        existing = {}

        conn = sqlite3.connect(DB_PATH)
        conn.execute(f"UPDATE metadata SET {DB_COLUMN} = 0")
        conn.commit()
        conn.close()
        print(f"   Reset all {DB_COLUMN} flags.")

if QDRANT_COLLECTION not in existing:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.EUCLID,
            on_disk=True,
        ),
    )
    print(f"✅ Created Qdrant collection '{QDRANT_COLLECTION}' "
          f"(size={VECTOR_SIZE}, Euclidean, on_disk).")

# ── Feature computation ───────────────────────────────────────────────────────

def compute_hog_feature(img: Image.Image) -> np.ndarray:
    img_gray    = img.convert("L")
    img_resized = img_gray.resize((128, 128))
    img_arr     = np.array(img_resized)

    features = hog(
        img_arr,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    grad_mag = np.std(img_arr.astype(np.float32))
    return (features * grad_mag).astype(np.float32)


def compute_colour_feature(img: Image.Image) -> np.ndarray:
    img_small = img.convert("RGB").resize((8, 8), Image.LANCZOS)
    arr_rgb   = np.array(img_small, dtype=np.float32) / 255.0
    arr_lab   = rgb2lab(arr_rgb)
    return arr_lab.flatten().astype(np.float32)


def fuse_batch(hog_vecs: np.ndarray, colour_vecs: np.ndarray) -> np.ndarray:
    """
    Concatenate HOG + Colour with configured weights.
    We NO LONGER L2-normalise here, as we want to preserve the 'edge strength'
    (magnitude) of the HOG features to distinguish flat vs textured patches.
    """
    return np.concatenate([
        FUSION_HOG_WEIGHT    * hog_vecs,
        FUSION_COLOUR_WEIGHT * colour_vecs,
    ], axis=1).astype(np.float32)

# ── Database helpers ──────────────────────────────────────────────────────────

def get_unindexed_images():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    if not c.fetchone():
        conn.close()
        return []
    c.execute(
        f"SELECT id, file_path, lat, lon, date FROM metadata WHERE {DB_COLUMN} = 0"
    )
    rows = c.fetchall()
    conn.close()
    return rows


def mark_as_indexed(ids):
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        f"UPDATE metadata SET {DB_COLUMN} = 1 WHERE id = ?", [(i,) for i in ids]
    )
    conn.commit()
    conn.close()

# ── Optional PCA — first pass to fit model ───────────────────────────────────

def fit_pca(rows):
    from sklearn.decomposition import PCA
    import joblib

    print(f"PCA_ENABLED=True but no model found at {PCA_PATH}.")
    print(f"Running first pass over {len(rows)} images to fit PCA({PCA_DIM})...")

    base_dir  = os.path.dirname(__file__)
    all_fused = []

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="PCA pass"):
        batch    = rows[i : i + BATCH_SIZE]
        hog_vecs = []
        col_vecs = []
        for _, rel_path, *_ in batch:
            abs_path = os.path.join(base_dir, rel_path)
            try:
                img = Image.open(abs_path).convert("RGB")
                hog_vecs.append(compute_hog_feature(img))
                col_vecs.append(compute_colour_feature(img))
            except Exception as e:
                print(f"  Warning: could not open {abs_path}: {e}")
                hog_vecs.append(np.zeros(HOG_VECTOR_SIZE, dtype=np.float32))
                col_vecs.append(np.zeros(COLOUR_VECTOR_SIZE, dtype=np.float32))
        all_fused.append(fuse_batch(np.array(hog_vecs), np.array(col_vecs)))

    X   = np.vstack(all_fused)
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(X)

    joblib.dump(pca, PCA_PATH)
    print(f"✅ PCA model saved to {PCA_PATH}  "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")
    return pca

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = get_unindexed_images()
    if not rows:
        print("✅ No unindexed images found. Everything is up to date!")
        qdrant.close()
        return

    print(f"Found {len(rows)} images to index with fused HOG+Colour hybrid.")

    pca_model = None
    if PCA_ENABLED:
        import joblib
        if not os.path.isfile(PCA_PATH):
            pca_model = fit_pca(rows)
        else:
            pca_model = joblib.load(PCA_PATH)
            print(f"✅ PCA model loaded from {PCA_PATH}")

    base_dir = os.path.dirname(__file__)

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Indexing"):
        batch       = rows[i : i + BATCH_SIZE]
        hog_vecs    = []
        col_vecs    = []
        valid_rows  = []

        for row in batch:
            id_, rel_path, lat, lon, date_str = row
            abs_path = os.path.join(base_dir, rel_path)
            try:
                img = Image.open(abs_path).convert("RGB")
                hog_vecs.append(compute_hog_feature(img))
                col_vecs.append(compute_colour_feature(img))
                valid_rows.append(row)
            except Exception as e:
                print(f"Failed to process {abs_path}: {e}")

        if not hog_vecs:
            continue

        fused = fuse_batch(np.array(hog_vecs), np.array(col_vecs))
        if pca_model is not None:
            fused = pca_model.transform(fused).astype(np.float32)

        points      = []
        ids_to_mark = []
        for j, row in enumerate(valid_rows):
            id_, rel_path, lat, lon, date_str = row
            points.append(
                PointStruct(
                    id=id_,
                    vector=fused[j].tolist(),
                    payload={
                        "filename": os.path.basename(rel_path),
                        "lat":      lat,
                        "lon":      lon,
                        "date":     date_str,
                    },
                )
            )
            ids_to_mark.append(id_)

        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        mark_as_indexed(ids_to_mark)

    print("✅ Indexing complete.")
    qdrant.close()


if __name__ == "__main__":
    main()