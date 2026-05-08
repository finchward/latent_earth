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

  const srcSide = Math.min(uploadedImage.width, uploadedImage.height);
  const sx = (uploadedImage.width - srcSide) / 2;
  const sy = (uploadedImage.height - srcSide) / 2;

  // Display size (max 500px)
  const displaySize = Math.min(srcSide, 500);
  canvas.width = displaySize;
  canvas.height = displaySize;

  // Draw image (center-cropped)
  ctx.drawImage(uploadedImage, sx, sy, srcSide, srcSide, 0, 0, displaySize, displaySize);

  // Grid overlay
  const effectiveSide = Math.min(srcSide, 2000);
  const n = Math.max(1, Math.floor(effectiveSide / pixelsPerPatch));
  const gridPx = displaySize / n;

  // White lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
  ctx.lineWidth = 1;
  for (let i = 1; i < n; i++) {
    const pos = Math.round(i * gridPx);
    ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, displaySize); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(displaySize, pos); ctx.stroke();
  }
  // Dark shadow lines for contrast
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
  for (let i = 1; i < n; i++) {
    const pos = Math.round(i * gridPx) + 1;
    ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, displaySize); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(displaySize, pos); ctx.stroke();
  }

  // Update info text
  $('grid-info').textContent = `${n} × ${n} grid · ${n * n} patches`;
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
  const effectiveSide = Math.min(
    Math.min(uploadedImage.width, uploadedImage.height), 2000
  );
  const n = Math.max(1, Math.floor(effectiveSide / pixelsPerPatch));
  const outSize = n * pixelsPerPatch * outputScale;
  $('output-info').textContent = `Output: ${outSize} × ${outSize} px`;
}

function updateConvertButton() {
  $('btn-convert').disabled = !(appReady && uploadedImage);
}

// ── Convert ────────────────────────────────────────────────────────────────
$('btn-convert').addEventListener('click', async () => {
  if (!uploadedImage || !appReady) return;

  // Re-encode at full resolution (center-cropped, capped at 2000)
  const srcSide = Math.min(uploadedImage.width, uploadedImage.height);
  const side = Math.min(srcSide, 2000);
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = side;
  tmpCanvas.height = side;
  const tctx = tmpCanvas.getContext('2d');

  const sx = (uploadedImage.width - srcSide) / 2;
  const sy = (uploadedImage.height - srcSide) / 2;
  tctx.drawImage(uploadedImage, sx, sy, srcSide, srcSide, 0, 0, side, side);

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
