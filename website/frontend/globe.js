// globe.js — Sentinel Atlas procedural globe

const $ = id => document.getElementById(id);

let appReady = false;

// ── Status polling ─────────────────────────────────────────────────────────
async function pollStatus() {
    try {
        const r = await fetch('/api/status');
        const s = await r.json();
        if (s.ready) {
            $('status-pill').textContent = `Ready · ${s.n_vectors.toLocaleString()} vectors`;
            $('status-pill').classList.add('ready');
            appReady = true;
            checkGrid(); // Trigger initial load
        } else {
            $('status-pill').textContent = s.message || 'Initialising…';
            setTimeout(pollStatus, 1800);
        }
    } catch (_) {
        setTimeout(pollStatus, 2500);
    }
}

// ── Hidden Canvas & Logic ──────────────────────────────────────────────────
const PATCH_SIZE = 256;
const GRID_SIZE = 9; // 9x9 buffer
const VIEW_SIZE = 5; // 5x5 visible window
const canvas = document.createElement('canvas');
canvas.width = GRID_SIZE * PATCH_SIZE;
canvas.height = GRID_SIZE * PATCH_SIZE;
const ctx = canvas.getContext('2d', { alpha: false });

// Initialize canvas with white (fallback)
ctx.fillStyle = '#ffffff';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// State
let cx = 0.0; // Continuous Cartesian X
let cy = 0.0; // Continuous Cartesian Y
let originX = Math.round(cx) - Math.floor(GRID_SIZE / 2);
let originY = Math.round(cy) - Math.floor(GRID_SIZE / 2);

const loadedImages = new Map(); // "x_y" -> HTMLImageElement
const pendingRequests = new Set(); // "x_y" strings currently being fetched

// Draw the current state of loaded images onto the canvas
function renderCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let x = originX; x < originX + GRID_SIZE; x++) {
        for (let y = originY; y < originY + GRID_SIZE; y++) {
            const key = `${x}_${y}`;
            const img = loadedImages.get(key);

            if (img && img !== 'failed') {
                const dx = x - originX;
                const dy = y - originY;

                // Map Cartesian (where +Y is UP) to Canvas (where Y=0 is TOP)
                const px = dx * PATCH_SIZE;
                const py = (GRID_SIZE - 1 - dy) * PATCH_SIZE;

                ctx.drawImage(img, px, py, PATCH_SIZE, PATCH_SIZE);
            }
        }
    }

    if (globeTexture) globeTexture.needsUpdate = true;
}

// Check which integer coordinates are missing in our 9x9 grid and batch request them
async function checkGrid() {
    if (!appReady) return;

    const currentGridCenterX = Math.round(cx);
    const currentGridCenterY = Math.round(cy);
    const newOriginX = currentGridCenterX - Math.floor(GRID_SIZE / 2);
    const newOriginY = currentGridCenterY - Math.floor(GRID_SIZE / 2);

    // If the origin shifted, we need to completely redraw the canvas buffer
    if (newOriginX !== originX || newOriginY !== originY) {
        originX = newOriginX;
        originY = newOriginY;
        renderCanvas();
    }

    const missingCoords = [];
    for (let x = originX; x < originX + GRID_SIZE; x++) {
        for (let y = originY; y < originY + GRID_SIZE; y++) {
            const key = `${x}_${y}`;
            if (!loadedImages.has(key) && !pendingRequests.has(key)) {
                missingCoords.push({ x, y });
                pendingRequests.add(key);
            }
        }
    }

    if (missingCoords.length === 0) return;

    try {
        const res = await fetch('/api/globe-patches', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coords: missingCoords })
        });

        if (!res.ok) throw new Error("Batch request failed");

        const data = await res.json();

        // Process incoming image URLs
        for (const [key, url] of Object.entries(data)) {
            if (url) {
                const img = new Image();
                img.crossOrigin = "anonymous";
                img.onload = () => {
                    loadedImages.set(key, img);
                    pendingRequests.delete(key);
                    renderCanvas();
                };
                img.onerror = () => {
                    loadedImages.set(key, 'failed');
                    pendingRequests.delete(key);
                };
                img.src = url;
            } else {
                loadedImages.set(key, 'failed');
                pendingRequests.delete(key);
            }
        }
    } catch (err) {
        console.error("[Globe] Error fetching patches:", err);
        missingCoords.forEach(c => pendingRequests.delete(`${c.x}_${c.y}`));
    }
}

// ── Three.js Setup ─────────────────────────────────────────────────────────
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();

// Camera
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.z = 6.5;

// Renderer
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

// Texture
const globeTexture = new THREE.CanvasTexture(canvas);
globeTexture.minFilter = THREE.LinearFilter;
globeTexture.magFilter = THREE.LinearFilter;

// Geometry: A 90x90 degree horizontal & vertical slice of a sphere 
// This creates a perfectly curved "dome" facing the camera without pole-pinching
const geometry = new THREE.SphereGeometry(
    5,               // Radius
    64, 64,          // Segments
    Math.PI * 0.25,  // phiStart
    Math.PI * 0.5,   // phiLength (90 degrees width)
    Math.PI * 0.25,  // thetaStart
    Math.PI * 0.5    // thetaLength (90 degrees height)
);

const material = new THREE.MeshBasicMaterial({ map: globeTexture });
const sphereDome = new THREE.Mesh(geometry, material);
scene.add(sphereDome);

// ── Interaction Logic ──────────────────────────────────────────────────────
let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };

function onPointerDown(e) {
    isDragging = true;
    previousMousePosition = { x: e.clientX, y: e.clientY };
}
const sens = 0.020;
function onPointerMove(e) {
    if (!isDragging) return;

    const deltaX = e.clientX - previousMousePosition.x;
    const deltaY = e.clientY - previousMousePosition.y;

    // Update continuous coordinates
    // Dragging left (-deltaX) means we look right (+cx)
    cx -= deltaX * sens;
    cy += deltaY * sens; // Dragging up (-deltaY) means we look down (-cy)

    previousMousePosition = { x: e.clientX, y: e.clientY };

    $('coord-display').textContent = `X: ${cx.toFixed(2)} | Y: ${cy.toFixed(2)}`;

    updateUVs();
    checkGrid();
}

function onPointerUp() {
    isDragging = false;
}

container.addEventListener('pointerdown', onPointerDown);
window.addEventListener('pointermove', onPointerMove);
window.addEventListener('pointerup', onPointerUp);

// Apply mathematical offsets to the texture
function updateUVs() {
    // Bottom-left of the visible 5x5 window
    const visX = cx - (VIEW_SIZE / 2);
    const visY = cy - (VIEW_SIZE / 2);

    // Calculate relative offset against the bottom-left of our 9x9 Canvas buffer
    const offsetX = (visX - originX) / GRID_SIZE;
    const offsetY = (visY - originY) / GRID_SIZE;

    globeTexture.offset.set(offsetX, offsetY);
    globeTexture.repeat.set(VIEW_SIZE / GRID_SIZE, VIEW_SIZE / GRID_SIZE);
}

// ── Resize handler & Render Loop ───────────────────────────────────────────
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

// ── Init ───────────────────────────────────────────────────────────────────
updateUVs();
animate();
pollStatus();