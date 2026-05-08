import os
import sqlite3
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm.auto import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/dinov2-small"
VECTOR_SIZE = 384
QDRANT_COLLECTION = "satellite_patches_dinov2"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "metadata.sqlite")
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_dinov2")

# ── Init ─────────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE}")
print(f"Loading {MODEL_NAME}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

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
        # Reset indexed_dinov2 flags so all images get re-indexed
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE metadata SET indexed_dinov2 = 0")
        conn.commit()
        conn.close()
        print("   Reset all indexed_dinov2 flags.")

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
    
    c.execute("SELECT id, file_path, lat, lon, date FROM metadata WHERE indexed_dinov2 = 0")
    rows = c.fetchall()
    conn.close()
    return rows

def mark_as_indexed(ids):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany("UPDATE metadata SET indexed_dinov2 = 1 WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()

def main():
    rows = get_unindexed_images()
    if not rows:
        print("✅ No unindexed images found. Everything is up to date!")
        return

    print(f"Found {len(rows)} images to index with DINOv2.")
    
    base_dir = os.path.dirname(__file__)
    
    for i in tqdm(range(0, len(rows), BATCH_SIZE)):
        batch = rows[i:i + BATCH_SIZE]
        
        images = []
        valid_batch = []
        for row in batch:
            id_, rel_path, lat, lon, date_str = row
            abs_path = os.path.join(base_dir, rel_path)
            try:
                img = Image.open(abs_path).convert("RGB")
                images.append(img)
                valid_batch.append(row)
            except Exception as e:
                print(f"Failed to process image {abs_path}: {e}")
        
        if not images:
            continue

        # Embed
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings = features.cpu().numpy().astype(np.float32)

        # Upsert
        points = []
        ids_to_mark = []
        for j, row in enumerate(valid_batch):
            id_, rel_path, lat, lon, date_str = row
            points.append(
                PointStruct(
                    id=id_,
                    vector=embeddings[j].tolist(),
                    payload={
                        "filename": os.path.basename(rel_path),
                        "lat": lat,
                        "lon": lon,
                        "date": date_str
                    }
                )
            )
            ids_to_mark.append(id_)
        
        if points:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
            mark_as_indexed(ids_to_mark)

    print("✅ Indexing complete.")
    qdrant.close()

if __name__ == "__main__":
    main()
