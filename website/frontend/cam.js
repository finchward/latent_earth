// cam.js — Sentinel Atlas frontend (camera only)

// ── Globals ────────────────────────────────────────────────────────────────
let appReady = false;
const $ = id => document.getElementById(id);

// ── Status polling ─────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    if (s.ready) {
      $('status-pill').textContent = `Ready · ${s.n_vectors.toLocaleString()} vectors`;
      $('status-pill').classList.add('ready');
      $('n-vectors').textContent = s.n_vectors.toLocaleString();
      $('btn-start-camera').disabled = false;
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
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const div = document.createElement('div');
      div.className = 'idle-cell' + (r === 3 && c === 3 ? ' ic-center' : '');
      g.appendChild(div);
    }
  }
}

// ── Show grid ──────────────────────────────────────────────────────────────
function showGrid() {
  $('idle-wrapper').style.display = 'none';
  $('grid-outer').style.display   = 'block';
  $('grid-outer').classList.add('fade-in');
}

// ── Camera ─────────────────────────────────────────────────────────────────
let cameraStream = null;
let cameraInterval = null;
let cameraGridInitialized = false;
let isProcessingFrame = false;

const videoEl  = $('camera-video');
const canvasEl = $('camera-canvas');
const ctx      = canvasEl ? canvasEl.getContext('2d') : null;

$('btn-start-camera').addEventListener('click', async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoEl.srcObject = cameraStream;
    $('btn-start-camera').disabled = true;
    $('btn-stop-camera').disabled  = false;

    if (!cameraGridInitialized) {
      initCameraGrid();
      cameraGridInitialized = true;
    }
    showGrid();

    cameraInterval = setInterval(processCameraFrame, 150);
  } catch (err) {
    alert('Camera error: ' + err.message);
  }
});

$('btn-stop-camera').addEventListener('click', () => {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
  }
  clearInterval(cameraInterval);
  $('btn-start-camera').disabled = false;
  $('btn-stop-camera').disabled  = true;
});

function initCameraGrid() {
  const wrapper = $('grid-wrapper');
  wrapper.innerHTML = '';

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell  = document.createElement('div');
      const inner = document.createElement('div');
      cell.className  = 'cell';
      inner.className = 'cell-inner';

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

async function processCameraFrame() {
  if (isProcessingFrame) return;
  isProcessingFrame = true;
  try {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (vw === 0 || vh === 0) { isProcessingFrame = false; return; }

    const size = Math.min(vw, vh);
    const sx = (vw - size) / 2;
    const sy = (vh - size) / 2;

    ctx.drawImage(videoEl, sx, sy, size, size, 0, 0, 800, 800);
    const b64 = canvasEl.toDataURL('image/jpeg', 0.8).split(',')[1];

    const resp = await fetch('/api/camera-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame_b64: b64 }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);

    const hostUrl = window.location.origin;
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const img = $(`cam-img-${r}-${c}`);
        const url = data.urls[r][c];
        if (!url) continue;
        const fullUrl = url.startsWith('/') ? hostUrl + url : url;
        if (img.src !== fullUrl) {
          img.src = fullUrl;
          img.style.opacity = '1';
        }
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
