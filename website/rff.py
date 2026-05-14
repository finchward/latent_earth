"""
rff.py
Multi-scale Random Fourier Feature mapping: R^2 -> R^N
Used to generate continuous latent vectors for the procedural globe.
"""

import os
import numpy as np
import joblib
import config

INPUT_SCALING = 0.5

class RFFMapper:
    def __init__(self, out_dim: int, k_bands: int = 6, seed: int = 42):
        self.out_dim = out_dim
        self.k_bands = k_bands
        
        # Deterministic random state so the globe layout never changes between reloads
        rng = np.random.RandomState(seed)
        
        # 1. Random 2D unit direction vectors w_ik
        angles = rng.uniform(0, 2 * np.pi, size=(out_dim, k_bands))
        self.W = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # Shape: (out_dim, k_bands, 2)
        
        # 2. Random phase offsets
        self.Phi = rng.uniform(0, 2 * np.pi, size=(out_dim, k_bands)) # Shape: (out_dim, k_bands)
        
        # 3. Amplitude that decays with frequency (0.5^k)
        self.A = 0.5 ** np.arange(k_bands)                            # Shape: (k_bands,)
        
        # 4. Frequency scale (2^k)
        self.freqs = 2.0 ** np.arange(k_bands)                        # Shape: (k_bands,)

        # 5. PCA-based scaling factors
        # We load the PCA model to understand the "shape" of our satellite embeddings.
        # By scaling each dimension by sqrt(explained_variance), we ensure our procedural
        # latent vectors land within the actual data distribution (the hyper-ellipsoid).
        self.scales = np.ones(out_dim, dtype=np.float32)
        if config.PCA_ENABLED and os.path.isfile(config.PCA_MODEL_PATH):
            try:
                pca = joblib.load(config.PCA_MODEL_PATH)
                # Ensure the PCA model dimensions match our expected out_dim
                if len(pca.explained_variance_) >= out_dim:
                    self.scales = np.sqrt(pca.explained_variance_[:out_dim]).astype(np.float32)
                    print(f"[RFF] Loaded PCA scales from {config.PCA_MODEL_PATH}")
                else:
                    print(f"[RFF] Warning: PCA model has only {len(pca.explained_variance_)} components, expected {out_dim}.")
            except Exception as e:
                print(f"[RFF] Error loading PCA scales: {e}")

    def map_points(self, points: list[tuple[float, float]]) -> np.ndarray:
        """
        Maps a list of 2D (x,y) coordinates to PCA-aligned out_dim-dimensional vectors.
        """
        if not points:
            return np.empty((0, self.out_dim), dtype=np.float32)
            
        P = np.array(points, dtype=np.float32)  # Shape: (N, 2)
        
        # Calculate dot products W_ik * p -> Shape: (N, out_dim, k_bands)
        dot_prod = np.einsum('dkc,nc->ndk', self.W, P) * INPUT_SCALING
        
        # Apply frequency scaling 2^k -> Shape: (N, out_dim, k_bands)
        scaled_dot = dot_prod * self.freqs
        
        # Add phase and compute sine -> Shape: (N, out_dim, k_bands)
        sin_vals = np.sin(scaled_dot + self.Phi)
        
        # Multiply by amplitude and sum across frequency bands -> Shape: (N, out_dim)
        z = np.sum(sin_vals * self.A, axis=-1)
        
        # SCALE by PCA components instead of normalizing to a unit sphere.
        # This aligns the procedural noise with the real data distribution.
        z = z * self.scales
        
        return z.astype(np.float32)

# Singleton to hold the initialized mapper
_mapper = None

def get_mapper() -> RFFMapper:
    global _mapper
    if _mapper is None:
        # We align the output dimension to match the chosen Qdrant vector size
        # (Defaults to 256 if config uses Fused Hybrid with PCA_DIM=256)
        _mapper = RFFMapper(out_dim=config.VECTOR_SIZE)
    return _mapper