import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, Prefetch, FusionQuery, Fusion

client = QdrantClient(path="../data_collection/data/qdrant_hog_hybrid")

def search(color_vec, name):
    hog = np.zeros((1568,), dtype=np.float32).tolist()
    
    print(f"--- Searching {name} ---")
    
    # colour only
    res_color = client.query_points(
        collection_name="satellite_patches_hog_hybrid",
        query=color_vec,
        using="colour",
        limit=3,
        with_payload=True
    )
    print("Colour only:")
    for p in res_color.points:
        print(f"  {p.payload['filename']} (score: {p.score})")

    # hog only
    res_hog = client.query_points(
        collection_name="satellite_patches_hog_hybrid",
        query=hog,
        using="hog",
        limit=3,
        with_payload=True
    )
    print("HOG only:")
    for p in res_hog.points:
        print(f"  {p.payload['filename']} (score: {p.score})")


    # Fusion DBSF
    res_fusion = client.query_points(
        collection_name="satellite_patches_hog_hybrid",
        prefetch=[
            Prefetch(query=hog, using="hog", limit=1000),
            Prefetch(query=color_vec, using="colour", limit=1000),
        ],
        query=FusionQuery(fusion=Fusion.DBSF),
        limit=3,
        with_payload=True
    )
    print("Fusion DBSF:")
    for p in res_fusion.points:
        print(f"  {p.payload['filename']} (score: {p.score})")


white_vec = np.full((192,), 1.0, dtype=np.float32).tolist()
search(white_vec, "White (1.0, 1.0, 1.0)")

blue_vec = np.array([[0.53, 0.8, 0.92] for _ in range(64)]).flatten().tolist()
search(blue_vec, "Blue (0.53, 0.8, 0.92)")
