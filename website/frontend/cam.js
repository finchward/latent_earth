// cam.js — Sentinel Atlas frontend (camera only)

// ── Globals ────────────────────────────────────────────────────────────────
let appReady = false;
let camPatchDim = 12;
let camUpdateMs = 1000;
let cameraStream = null;
let cameraInterval = null;
let cameraGridInitialized = false;
let isProcessingFrame = false;
let cameraStarted = false;
let ctx = null;
let randomTextInterval = null;
let audioStarted = false;

const $ = id => document.getElementById(id);

// Helper to generate a random 20-40 char string of upper/lowercase letters
function generateRandomString() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
  const length = Math.floor(Math.random() * 15) + 30; // 20 to 40 inclusive
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

// ── Status polling ─────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    if (s.ready) {
      $('status-pill').textContent = `Ready · ${s.n_vectors.toLocaleString()} vectors`;
      $('status-pill').classList.add('ready');
      $('n-vectors').textContent = s.n_vectors.toLocaleString();

      if (s.cam_patch_dim && s.cam_patch_dim !== camPatchDim) {
        camPatchDim = s.cam_patch_dim;
        buildIdleGrid(); // Rebuild if dimension changed
      }

      if (s.cam_update_ms) {
        camUpdateMs = s.cam_update_ms;
      }

      appReady = true;
    } else {
      $('status-pill').textContent = s.message || 'Initialising…';
      setTimeout(pollStatus, 1800);
    }
  } catch (_) {
    setTimeout(pollStatus, 2500);
  }
}

// ── Idle placeholder grid ──────────────────────────────────────────────────
function buildIdleGrid() {
  const g = $('idle-grid');
  if (!g) return;
  g.innerHTML = '';

  const cellSize = window.innerHeight / camPatchDim;
  g.style.gridTemplateColumns = `repeat(${camPatchDim}, ${cellSize}px)`;
  g.style.gridTemplateRows = `repeat(${camPatchDim}, ${cellSize}px)`;

  const mid = Math.floor(camPatchDim / 2);

  for (let r = 0; r < camPatchDim; r++) {
    for (let c = 0; c < camPatchDim; c++) {
      const div = document.createElement('div');
      div.className = 'idle-cell' + (r === mid && c === mid ? ' ic-center' : '');
      div.style.width = `${cellSize}px`;
      div.style.height = `${cellSize}px`;
      g.appendChild(div);
    }
  }
}

// ── Resize handling ────────────────────────────────────────────────────────
function handleResize() {
  const cellSize = window.innerHeight / camPatchDim;

  const grids = [$('idle-grid'), $('grid-wrapper')];
  grids.forEach(g => {
    if (!g) return;
    // In initCameraGrid() and handleResize():
    g.style.gridTemplateColumns = `repeat(${camPatchDim}, 1fr)`;
    g.style.gridTemplateRows = `repeat(${camPatchDim}, 1fr)`;

    // Update individual cells
    const cells = g.querySelectorAll('.cell, .idle-cell');
    cells.forEach(cell => {
      cell.style.width = `${cellSize}px`;
      cell.style.height = `${cellSize}px`;
    });
  });
}
window.addEventListener('resize', handleResize);

// ── Show grid ──────────────────────────────────────────────────────────────
function showGrid() {
  const wrapper = $('idle-wrapper');
  const outer = $('grid-outer');
  if (wrapper) wrapper.style.display = 'none';
  if (outer) {
    outer.style.display = 'block';
    outer.classList.add('fade-in');
  }
}

// ── Camera ─────────────────────────────────────────────────────────────────
async function startCamera() {
  if (cameraStarted) return;
  console.log("[Camera] Attempting to start...");

  const v = $('camera-video');
  const c = $('camera-canvas');
  if (!v || !c) {
    console.error("[Camera] video or canvas element not found!");
    return;
  }

  if (!ctx) ctx = c.getContext('2d');

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    v.srcObject = cameraStream;
    await v.play();
    console.log("[Camera] Stream started and playing.");

    cameraStarted = true;

    if (!cameraGridInitialized) {
      initCameraGrid();
      cameraGridInitialized = true;
    }
    showGrid();

    console.log(`[Camera] Starting update interval: ${camUpdateMs}ms`);
    cameraInterval = setInterval(processCameraFrame, camUpdateMs);

    // Hide the overlay on success
    const overlay = $('camera-overlay');
    if (overlay) overlay.classList.add('hidden');

  } catch (err) {
    console.warn('[Camera] Start blocked or failed:', err);
    cameraStarted = false;
    $('status-pill').textContent = "Camera Error — Click to retry";
    $('status-pill').classList.add('error');
  }
}

function initCameraGrid() {
  const wrapper = $('grid-wrapper');
  if (!wrapper) return;
  wrapper.innerHTML = '';

  const cellSize = window.innerHeight / camPatchDim;
  // In initCameraGrid() and handleResize():
  wrapper.style.gridTemplateColumns = `repeat(${camPatchDim}, 1fr)`;
  wrapper.style.gridTemplateRows = `repeat(${camPatchDim}, 1fr)`;

  for (let r = 0; r < camPatchDim; r++) {
    for (let c = 0; c < camPatchDim; c++) {
      const cell = document.createElement('div');
      const inner = document.createElement('div');
      cell.className = 'cell';
      inner.className = 'cell-inner';
      inner.id = `cam-inner-${r}-${c}`;

      cell.style.width = `${cellSize}px`;
      cell.style.height = `${cellSize}px`;

      const img = document.createElement('img');
      img.id = `cam-img-${r}-${c}`;
      img.style.opacity = '0';
      img.style.transition = 'opacity 0.3s';
      inner.appendChild(img);

      cell.appendChild(inner);
      wrapper.appendChild(cell);
    }
  }
}

// --- Start audio if it hasn't started yet now that patches are loaded ---
if (!audioStarted) {
  const sonarAudio = $('sonar-audio');
  if (sonarAudio) {
    sonarAudio.play().catch(err => {
      console.warn('[Audio] Playback blocked or failed:', err);
    });
    audioStarted = true;
  }
}

async function processCameraFrame() {
  if (isProcessingFrame) return;
  isProcessingFrame = true;
  try {
    const v = $('camera-video');
    const c = $('camera-canvas');
    if (!v || !c || !ctx) {
      isProcessingFrame = false;
      return;
    }

    const vw = v.videoWidth;
    const vh = v.videoHeight;
    if (vw === 0 || vh === 0 || v.readyState < 2) {
      isProcessingFrame = false;
      return;
    }

    const size = Math.min(vw, vh);
    const sx = (vw - size) / 2;
    const sy = (vh - size) / 2;
    ctx.drawImage(v, sx, sy, size, size, 0, 0, 400, 400);

    const blob = await new Promise(resolve => c.toBlob(resolve, 'image/jpeg', 0.6));
    if (!blob || blob.size < 100) {
      isProcessingFrame = false;
      return;
    }

    const resp = await fetch('/api/camera-frame', {
      method: 'POST',
      body: blob,
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);

    const hostUrl = window.location.origin;
    for (let r = 0; r < camPatchDim; r++) {
      for (let c = 0; c < camPatchDim; c++) {
        const img = $(`cam-img-${r}-${c}`);
        if (!img) continue;
        const url = data.urls[r][c];
        if (!url) continue;
        const fullUrl = url.startsWith('/') ? hostUrl + url : url;
        if (img.src !== fullUrl) {
          img.src = fullUrl;
          img.style.opacity = '1';
        }

        const meta = data.meta ? data.meta[r][c] : null;
        if (meta) {
          const inner = $(`cam-inner-${r}-${c}`);
          if (inner) {
            inner.dataset.lat = meta.lat;
            inner.dataset.lon = meta.lon;
            inner.dataset.date = meta.date;
          }
        }
      }
    }


    // ------------------------------------------------------------------------

    // Refresh tooltip if we are currently hovering
    if (typeof currentHoveredInner !== 'undefined' && currentHoveredInner) {
      const lat = currentHoveredInner.dataset.lat;
      const lon = currentHoveredInner.dataset.lon;
      const date = currentHoveredInner.dataset.date;
      if (lat && lon) {
        const tooltip = $('gtooltip');
        tooltip.innerHTML = `${parseFloat(lat).toFixed(4)}, ${parseFloat(lon).toFixed(4)}`;
        // Note: left/top are already set by the last mousemove event
        tooltip.style.display = 'block';
      } else {
        $('gtooltip').style.display = 'none';
      }
    }

  } catch (err) {
    console.error('[Camera] Error:', err);
  } finally {
    isProcessingFrame = false;
  }
}

// ── Init ───────────────────────────────────────────────────────────────────
buildIdleGrid();
pollStatus();

// ── Tooltip & Sidebar logic ────────────────────────────────────────────────
let currentHoveredInner = null;
let geocodeTimeout = null;
let lastHoveredCoords = { lat: null, lon: null };

// Helper to format date and time to "Wednesday, June 6, 2020, 11:32pm"
function formatDateString(dateString) {
  if (!dateString) return 'Unknown Date';
  const d = new Date(dateString);
  if (isNaN(d)) return dateString; // fallback if invalid

  const formattedDate = d.toLocaleString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });

  return formattedDate
    .replace(' at ', ', ')
    .replace(' AM', 'am')
    .replace(' PM', 'pm');
}

function updateSidebarEmptyState(isEmpty) {
  const info = $('hover-info');

  if (isEmpty) {
    info.style.display = 'none';
    if (randomTextInterval) {
      clearInterval(randomTextInterval);
      $('info-location').classList.remove('scrambling'); // Remove class on empty
    }
  } else {
    info.style.display = 'flex';
  }
}

const gridWrapper = $('grid-wrapper');
if (gridWrapper) {
  gridWrapper.addEventListener('mousemove', (e) => {
    let target = e.target.closest('.cell-inner');
    if (target) {
      currentHoveredInner = target;
      const lat = target.dataset.lat;
      const lon = target.dataset.lon;
      const date = target.dataset.date;

      if (lat && lon) {
        // 1. Update floating tooltip
        const tooltip = $('gtooltip');
        tooltip.innerHTML = `${parseFloat(lat).toFixed(4)}, ${parseFloat(lon).toFixed(4)}`;
        tooltip.style.left = e.clientX + 'px';
        tooltip.style.top = e.clientY + 'px';
        tooltip.style.display = 'block';

        // 2. Update Sidebar Basic Info
        updateSidebarEmptyState(false);

        // 3. Fetch Location Name (Debounced by 400ms)
        if (lastHoveredCoords.lat !== lat || lastHoveredCoords.lon !== lon) {
          lastHoveredCoords = { lat, lon };

          if (randomTextInterval) clearInterval(randomTextInterval);

          // Add the scrambling class right before starting the interval
          $('info-location').classList.add('scrambling');

          randomTextInterval = setInterval(() => {
            $('info-location').textContent = generateRandomString();
          }, 10);

          clearTimeout(geocodeTimeout);
          geocodeTimeout = setTimeout(async () => {
            try {
              const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10`);
              const data = await res.json();

              // Stop animation and remove class when data arrives
              if (randomTextInterval) clearInterval(randomTextInterval);
              $('info-location').classList.remove('scrambling');

              if (data && data.address) {
                const addr = data.address;
                const place = addr.city || addr.town || addr.municipality || addr.county || addr.state || "Unknown Area";
                const country = addr.country || "";
                $('info-location').textContent = `${place}${country ? ' in ' + country : ''}${' on ' + formatDateString(date)}`;
              } else {
                $('info-location').textContent = "Unknown Location";
              }
            } catch (err) {
              console.error('[Geocode] Error:', err);
              // Stop animation and remove class on error
              if (randomTextInterval) clearInterval(randomTextInterval);
              $('info-location').classList.remove('scrambling');
              $('info-location').textContent = "Location unavailable";
            }
          }, 400);
        }

      } else {
        $('gtooltip').style.display = 'none';
        updateSidebarEmptyState(true);
      }
    } else {
      currentHoveredInner = null;
      $('gtooltip').style.display = 'none';
      updateSidebarEmptyState(true);
    }
  });

  gridWrapper.addEventListener('mouseleave', () => {
    currentHoveredInner = null;
    $('gtooltip').style.display = 'none';
    updateSidebarEmptyState(true);
  });
}

// ── Event Listeners ────────────────────────────────────────────────────────
document.addEventListener('click', () => {
  if (appReady && !cameraStarted) {
    startCamera();
  }
}, { once: true });

const enableBtn = $('btn-enable-camera');
if (enableBtn) {
  enableBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    startCamera();
  });
}