// convert.js — Sentinel Atlas convert page

const $ = id => document.getElementById(id);

// ── State ──────────────────────────────────────────────────────────────────
let appReady = false;
let uploadedImage = null;   // HTMLImageElement
let outputScale = 2;
let pixelsPerPatch = 100;

// ── Status polling ─────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    if (s.ready) {
      $('status-pill').textContent = `Ready · ${s.n_vectors.toLocaleString()} vectors`;
      $('status-pill').classList.add('ready');
      appReady = true;
      updateConvertButton();
    } else {
      $('status-pill').textContent = s.message || 'Initialising…';
      setTimeout(pollStatus, 1800);
    }
  } catch (_) {
    setTimeout(pollStatus, 2500);
  }
}

// ── Upload zone ────────────────────────────────────────────────────────────
const uploadZone = $('upload-zone');
const fileInput = $('file-input');

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      uploadedImage = img;
      onImageLoaded();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function onImageLoaded() {
  // Show settings + workspace
  $('settings-panel').style.display = '';
  $('empty-state').style.display = 'none';
  $('convert-workspace').style.display = '';
  $('output-panel').style.display = 'none';

  // Update upload zone text
  uploadZone.querySelector('.upload-text').textContent = 'Change image';
  uploadZone.querySelector('.upload-sub').textContent =
    `${uploadedImage.width} × ${uploadedImage.height} px`;

  updatePreview();
  updateConvertButton();
}

// ── Preview canvas with grid overlay ───────────────────────────────────────
function updatePreview() {
  if (!uploadedImage) return;

  const canvas = $('preview-canvas');
  const ctx = canvas.getContext('2d');

  // Cap resolution logic (same as backend)
  const MAX_RESOLUTION = 2000;
  let w = uploadedImage.width;
  let h = uploadedImage.height;
  if (w > MAX_RESOLUTION || h > MAX_RESOLUTION) {
    const scale = MAX_RESOLUTION / Math.max(w, h);
    w = Math.floor(w * scale);
    h = Math.floor(h * scale);
  }

  // Display scale (fit within 500x500)
  const displayScale = Math.min(1, 500 / Math.max(w, h));
  const dw = Math.floor(w * displayScale);
  const dh = Math.floor(h * displayScale);
  canvas.width = dw;
  canvas.height = dh;

  // Draw full image
  ctx.drawImage(uploadedImage, 0, 0, dw, dh);

  // Grid overlay
  const cols = Math.max(1, Math.floor(w / pixelsPerPatch));
  const rows = Math.max(1, Math.floor(h / pixelsPerPatch));

  const gridW = cols * pixelsPerPatch;
  const gridH = rows * pixelsPerPatch;
  
  // Display offsets
  const ox = ((w - gridW) / 2) * displayScale;
  const oy = ((h - gridH) / 2) * displayScale;
  const gapW = pixelsPerPatch * displayScale;
  const gapH = pixelsPerPatch * displayScale;

  // Overlay for cropped areas
  ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
  ctx.fillRect(0, 0, dw, oy); // Top
  ctx.fillRect(0, oy + gridH * displayScale, dw, dh - (oy + gridH * displayScale)); // Bottom
  ctx.fillRect(0, oy, ox, gridH * displayScale); // Left
  ctx.fillRect(ox + gridW * displayScale, oy, dw - (ox + gridW * displayScale), gridH * displayScale); // Right

  // White grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= cols; i++) {
    const x = Math.round(ox + i * gapW);
    ctx.beginPath(); ctx.moveTo(x, oy); ctx.lineTo(x, oy + gridH * displayScale); ctx.stroke();
  }
  for (let i = 0; i <= rows; i++) {
    const y = Math.round(oy + i * gapH);
    ctx.beginPath(); ctx.moveTo(ox, y); ctx.lineTo(ox + gridW * displayScale, y); ctx.stroke();
  }

  // Update info text
  $('grid-info').textContent = `${cols} × ${rows} grid · ${cols * rows} patches`;
  updateOutputInfo();
}

// ── Settings controls ──────────────────────────────────────────────────────
const sliderPPP = $('slider-ppp');
sliderPPP.addEventListener('input', () => {
  pixelsPerPatch = parseInt(sliderPPP.value);
  $('val-ppp').textContent = pixelsPerPatch;
  updatePreview();
});

// Output scale buttons
document.querySelectorAll('.scale-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    outputScale = parseInt(btn.dataset.scale);
    updateOutputInfo();
  });
});

function updateOutputInfo() {
  if (!uploadedImage) return;
  const MAX_RESOLUTION = 2000;
  let w = uploadedImage.width;
  let h = uploadedImage.height;
  if (w > MAX_RESOLUTION || h > MAX_RESOLUTION) {
    const scale = MAX_RESOLUTION / Math.max(w, h);
    w = Math.floor(w * scale);
    h = Math.floor(h * scale);
  }
  const cols = Math.max(1, Math.floor(w / pixelsPerPatch));
  const rows = Math.max(1, Math.floor(h / pixelsPerPatch));
  
  const outW = cols * pixelsPerPatch * outputScale;
  const outH = rows * pixelsPerPatch * outputScale;
  $('output-info').textContent = `Output: ${outW} × ${outH} px`;
}

function updateConvertButton() {
  $('btn-convert').disabled = !(appReady && uploadedImage);
}

// ── Convert ────────────────────────────────────────────────────────────────
$('btn-convert').addEventListener('click', async () => {
  if (!uploadedImage || !appReady) return;

  // Re-encode at full resolution (maintaining aspect ratio, capped at 2000)
  const MAX_RESOLUTION = 2000;
  let w = uploadedImage.width;
  let h = uploadedImage.height;
  if (w > MAX_RESOLUTION || h > MAX_RESOLUTION) {
    const scale = MAX_RESOLUTION / Math.max(w, h);
    w = Math.floor(w * scale);
    h = Math.floor(h * scale);
  }
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = w;
  tmpCanvas.height = h;
  const tctx = tmpCanvas.getContext('2d');
  tctx.drawImage(uploadedImage, 0, 0, w, h);

  const b64 = tmpCanvas.toDataURL('image/png').split(',')[1];

  // Show loading
  $('loading-overlay').classList.add('vis');
  $('loading-msg').textContent = 'Converting…';
  $('btn-convert').disabled = true;

  try {
    const resp = await fetch('/api/convert-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: b64,
        pixels_per_patch: pixelsPerPatch,
        output_scale: outputScale,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || resp.statusText);

    // Display output
    $('output-image').src = `data:image/png;base64,${data.image_b64}`;
    $('output-panel').style.display = '';
    $('output-panel').classList.add('fade-in');
  } catch (err) {
    alert('Conversion failed: ' + err.message);
    console.error('[Convert] Error:', err);
  } finally {
    $('loading-overlay').classList.remove('vis');
    updateConvertButton();
  }
});

// ── Download ───────────────────────────────────────────────────────────────
$('btn-download').addEventListener('click', () => {
  const img = $('output-image');
  if (!img.src) return;
  const a = document.createElement('a');
  a.href = img.src;
  a.download = 'sentinel-atlas-mosaic.png';
  a.click();
});

// ── Init ───────────────────────────────────────────────────────────────────
pollStatus();
