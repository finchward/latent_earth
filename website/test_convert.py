import os
import base64
import io
import sys
from PIL import Image

from converter import process_uploaded_image
from state import app_state
from qdrant_client import QdrantClient
from config import QDRANT_LOCAL_PATH

app_state.qdrant = QdrantClient(path=QDRANT_LOCAL_PATH)

# Make a dummy white square on black background
img = Image.new("RGB", (400, 400), color="black")
# Draw a white square in top-right corner
for x in range(200, 400):
    for y in range(0, 200):
        img.putpixel((x, y), (255, 255, 255))

buf = io.BytesIO()
img.save(buf, format="PNG")
image_bytes = buf.getvalue()

try:
    res = process_uploaded_image(image_bytes, pixels_per_patch=100, output_scale=1)
    
    out_b64 = res["image_b64"]
    out_img = Image.open(io.BytesIO(base64.b64decode(out_b64)))
    out_img.save("test_out.png")
    print("Success, saved test_out.png")
except Exception as e:
    import traceback
    traceback.print_exc()
