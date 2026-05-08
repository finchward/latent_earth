import numpy as np
from PIL import Image

def test_grad_mag():
    # 1. Perfectly smooth with Gaussian noise (std=2, typical webcam noise)
    noise = np.random.normal(0, 2, (100, 100))
    img_smooth = np.clip(128 + noise, 0, 255).astype(np.float32)
    dx = np.diff(img_smooth, axis=1)
    dy = np.diff(img_smooth, axis=0)
    print("Smooth + Noise grad_mag:", np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

    # 2. Harsh edge (e.g. shoulder against wall)
    img_edge = np.ones((100, 100), dtype=np.float32) * 200
    img_edge[:, 50:] = 50 # harsh edge down the middle
    dx = np.diff(img_edge, axis=1)
    dy = np.diff(img_edge, axis=0)
    print("Harsh edge grad_mag:", np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

    # 3. Highly textured (e.g. hair, forest)
    noise_heavy = np.random.normal(0, 30, (100, 100))
    img_tex = np.clip(128 + noise_heavy, 0, 255).astype(np.float32)
    dx = np.diff(img_tex, axis=1)
    dy = np.diff(img_tex, axis=0)
    print("Texture grad_mag:", np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

if __name__ == "__main__":
    test_grad_mag()
