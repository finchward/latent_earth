// app.js — Sentinel Atlas frontend

// ── Globals ────────────────────────────────────────────────────────────────
let appReady  = false;
let lastCoord = null;   // { lat, lon }

const tooltip = document.getElementById('gtooltip');
const $       = id => document.getElementById(id);

// ── Loading overlay ────────────────────────────────────────────────────────
function showLoading(msg) {
  $('loading-msg').textContent = msg;
  $('loading-overlay').classList.add('vis');
}
function hideLoading() {
  $('loading-overlay').classList.remove('vis');
}

// ── Error display ──────────────────────────────────────────────────────────
function showError(msg) {
  const el = $('error-msg');
  el.textContent  = msg;
  el.style.display = 'block';
}
function clearError() {
  $('error-msg').style.display = 'none';
}

// ── Fetch wrapper ──────────────────────────────────────────────────────────
async function api(path, body) {
  const opts = {
    method:  body !== undefined ? 'POST' : 'GET',
    headers: {},
  };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r    = await fetch(path, opts);
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

// ── Idle placeholder ───────────────────────────────────────────────────────
function buildIdleGrid() {
  const g = $('idle-grid');
  for (let r = 0; r < 17; r++) {
    for (let c = 0; c < 17; c++) {
      const div = document.createElement('div');
      div.className = 'idle-cell' + (r === 8 && c === 8 ? ' ic-center' : '');
      g.appendChild(div);
    }
  }
}

// ── Status polling ─────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const s = await api('/api/status');
    if (s.ready) {
      $('status-pill').textContent = `Ready · ${s.n_vectors.toLocaleString()} vectors`;
      $('status-pill').classList.add('ready');
      $('n-vectors').textContent = s.n_vectors.toLocaleString();
      if (s.eigenvalues && s.eigenvalues.length === 4) {
        const total = s.eigenvalues.reduce((a, b) => a + b, 0);
        $('total-ev').textContent = (total * 100).toFixed(1);
        $('eigen-info').style.display = 'block';
      }
      setButtons(true);
      appReady = true;
    } else {
      $('status-pill').textContent = s.message || 'Initialising…';
      setTimeout(pollStatus, 1800);
    }
  } catch (_) {
    setTimeout(pollStatus, 2500);
  }
}

// ── Button state ───────────────────────────────────────────────────────────
function setButtons(on) {
  $('btn-random').disabled = !on;
  $('btn-fetch').disabled  = !on;
}

// ── Random coord ───────────────────────────────────────────────────────────
$('btn-random').addEventListener('click', async () => {
  clearError();
  showLoading('Sampling random land coordinate…');
  setButtons(false);
  try {
    const { lat, lon } = await api('/api/random-coords', {});
    $('coord-input').value = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    lastCoord = { lat, lon };
    $('preview-section').style.display = 'none';
    $('pc-section').style.display      = 'none';
  } catch (e) {
    showError(e.message);
  } finally {
    setButtons(true);
    hideLoading();
  }
});

// ── Fetch patch ────────────────────────────────────────────────────────────
$('btn-fetch').addEventListener('click', async () => {
  clearError();
  const raw = $('coord-input').value.trim();
  if (!raw) { showError('Enter coordinates first.'); return; }

  showLoading('Querying Sentinel-2 archive…');
  setButtons(false);
  $('preview-section').style.display = 'none';
  $('pc-section').style.display      = 'none';

  try {
    const d = await api('/api/fetch-patch', { coord_str: raw });
    lastCoord = { lat: d.lat, lon: d.lon };

    $('preview-img').src       = 'data:image/png;base64,' + d.preview_b64;
    $('meta-lat').textContent  = d.lat.toFixed(5);
    $('meta-lon').textContent  = d.lon.toFixed(5);
    $('meta-date').textContent = d.date;

    $('preview-section').style.display = 'block';
    $('preview-section').classList.add('fade-in');
  } catch (e) {
    showError(e.message);
  } finally {
    setButtons(true);
    hideLoading();
  }
});

// ── Build grid ─────────────────────────────────────────────────────────────
$('btn-build').addEventListener('click', async () => {
  if (!lastCoord) return;
  clearError();
  showLoading('Projecting embeddings onto principal components…');
  setButtons(false);
  $('btn-build').disabled = true;

  try {
    const data = await api('/api/build-grid', lastCoord);
    renderGrid(data);
    applyGlow(data.pc_weights);
    showPCBars(data.pc_weights);
  } catch (e) {
    showError(e.message);
  } finally {
    setButtons(true);
    $('btn-build').disabled = false;
    hideLoading();
  }
});

// ── Render grid ────────────────────────────────────────────────────────────
function renderGrid(data) {
  const wrapper = $('grid-wrapper');
  wrapper.innerHTML = '';
  const { grid, query } = data;

  for (let r = 0; r < 17; r++) {
    for (let c = 0; c < 17; c++) {
      const cell  = document.createElement('div');
      const inner = document.createElement('div');
      cell.className  = 'cell';
      inner.className = 'cell-inner';

      if (r === 8 && c === 8) {
        cell.classList.add('query-cell');
        if (query.thumb_b64) {
          const img = document.createElement('img');
          img.src = 'data:image/png;base64,' + query.thumb_b64;
          inner.appendChild(img);
        }
        attachTooltip(cell,
          `QUERY\n${fmtCoord(query.lat, query.lon)}\n${query.date || ''}`
        );
      } else {
        const patch = grid[r][c];
        if (patch) {
          if (patch.axis && patch.axis !== 'diag') {
            cell.classList.add('ax-' + patch.axis);
          }
          const img = document.createElement('img');
          img.src     = 'data:image/png;base64,' + patch.thumb_b64;
          img.loading = 'lazy';
          inner.appendChild(img);
          attachTooltip(cell,
            `${fmtCoord(patch.lat, patch.lon)}\n${patch.date}\n${patch.axis}`
          );
        } else {
          cell.classList.add('empty');
        }
      }

      cell.appendChild(inner);
      wrapper.appendChild(cell);
    }
  }

  $('idle-wrapper').style.display = 'none';
  $('grid-outer').style.display   = 'block';
  $('grid-outer').classList.add('fade-in');
}

function fmtCoord(lat, lon) {
  if (lat == null || lon == null) return '—';
  return `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
}

// ── Tooltip ────────────────────────────────────────────────────────────────
function attachTooltip(el, text) {
  el.addEventListener('mouseenter', () => {
    tooltip.style.display = 'block';
    tooltip.innerHTML = text.replace(/\n/g, '<br>');
  });
  el.addEventListener('mousemove', e => {
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top  = (e.clientY - 18) + 'px';
  });
  el.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });
}

// ── Directional glow ───────────────────────────────────────────────────────
// PC1 → shadow right, PC2 → up, PC3 → left, PC4 → down.
// Each weight is |q · PC_k| / Σ|q · PC_j|.
function applyGlow(weights) {
  const [w1, w2, w3, w4] = weights;
  const maxOff  = 100;
  const maxBlur = 60;

  function shadow(xOff, yOff, w) {
    const off   = (w * maxOff).toFixed(1);
    const blur  = (w * maxBlur).toFixed(1);
    const alpha = (w * 0.78).toFixed(2);
    return `${xOff(off)}px ${yOff(off)}px ${blur}px rgba(0,0,0,${alpha})`;
  }

  const shadows = [
    shadow(v => v,        _ => '0',    w1),   // right
    shadow(_ => '0',      v => `-${v}`, w2),  // up
    shadow(v => `-${v}`,  _ => '0',    w3),   // left
    shadow(_ => '0',      v => v,      w4),   // down
  ].join(', ');

  $('grid-wrapper').style.boxShadow = shadows;
}

// ── PC bars ────────────────────────────────────────────────────────────────
function showPCBars(weights) {
  $('pc-section').style.display = 'block';
  weights.forEach((w, i) => {
    $('bar' + (i + 1)).style.width    = (w * 100).toFixed(1) + '%';
    $('pct' + (i + 1)).textContent = (w * 100).toFixed(1) + '%';
  });
}

// ── Init ───────────────────────────────────────────────────────────────────
buildIdleGrid();
pollStatus();
