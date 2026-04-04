# ╔══════════════════════════════════════════════════════════════════╗
# ║  ECG Lead Generation  |  7 → 12 Leads  |  Streamlit App        ║
# ║  Upload an ECG image or signal file → reconstruct full 12-lead  ║
# ╚══════════════════════════════════════════════════════════════════╝

import os, io, warnings
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.signal import butter, sosfilt, resample
from scipy.ndimage import median_filter
import base64, json

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Lead Generator · 7→12",
    page_icon="🫀",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
FS          = 500
SEQ_LEN     = 2500
ALL_LEADS   = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
INPUT_LEADS = ['I','II','III','aVR','aVL','aVF','V1']
TARGET_LEADS= ['V2','V3','V4','V5','V6']
IN_IDX      = list(range(7))
TGT_IDX     = list(range(7, 12))

# ── Signal helpers ────────────────────────────────────────────────────────────
def bandpass(sig: np.ndarray, lo=0.5, hi=40.0, fs=500, order=4) -> np.ndarray:
    nyq = fs / 2.0
    sos = butter(order, [lo/nyq, hi/nyq], btype='band', output='sos')
    return sosfilt(sos, sig).astype(np.float32)

def znorm(sig: np.ndarray) -> np.ndarray:
    mu  = sig.mean(axis=-1, keepdims=True)
    std = sig.std(axis=-1,  keepdims=True)
    std[std < 1e-8] = 1.0
    return ((sig - mu) / std).astype(np.float32)

def preprocess_signals(signals: np.ndarray) -> np.ndarray:
    """Bandpass + z-score each lead. Input: [N_leads, L] or [L]."""
    if signals.ndim == 1:
        return znorm(bandpass(signals).reshape(1, -1))[0]
    return znorm(np.stack([bandpass(signals[i]) for i in range(len(signals))]))

# ── Red-grid renderer ─────────────────────────────────────────────────────────
def render_redgrid(sig12: np.ndarray,
                   lead_names=None,
                   fs: int = FS,
                   img_sz: tuple = (896, 672)) -> Image.Image:
    names = lead_names or ALL_LEADS
    n     = sig12.shape[0]
    L     = sig12.shape[1]
    t     = np.linspace(0, L / fs, L)

    fig = plt.figure(figsize=(16, 12), facecolor='white', dpi=56)
    gs  = gridspec.GridSpec(6, 2, hspace=0.50, wspace=0.30)

    for idx in range(min(n, 12)):
        r, c = divmod(idx, 2)
        ax   = fig.add_subplot(gs[r, c])
        ax.set_facecolor('#FFF5F5')
        ax.set_xticks(np.arange(0, t[-1] + 0.01, 0.20))
        ax.set_yticks(np.arange(-2, 2.1, 0.5))
        ax.set_xticks(np.arange(0, t[-1] + 0.01, 0.04), minor=True)
        ax.set_yticks(np.arange(-2, 2.1, 0.1), minor=True)
        ax.grid(True, which='major', color='#FF9999', linewidth=0.55, alpha=0.85)
        ax.grid(True, which='minor', color='#FFD5D5', linewidth=0.22, alpha=0.60)
        ax.plot(t, sig12[idx], 'k-', linewidth=0.75)
        ax.set_title(names[idx], fontsize=7, fontweight='bold', pad=1)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-2.5, 2.5)
        ax.tick_params(labelsize=4)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB').resize(img_sz, Image.LANCZOS)

# ── ECG Image Digitizer ───────────────────────────────────────────────────────
def _render_lead_standalone(signal_1d: np.ndarray, fs: int = FS) -> Image.Image:
    """Render a single lead as a clean borderless image for digitization."""
    L = len(signal_1d)
    t = np.linspace(0, L / fs, L)
    fig, ax = plt.subplots(figsize=(10, 2), facecolor='white', dpi=80)
    ax.plot(t, np.clip(signal_1d, -4, 4), 'k-', linewidth=1.5)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-4, 4)
    ax.set_facecolor('white')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def _digitize_lead_image(pil_img: Image.Image,
                          seq_len: int = SEQ_LEN,
                          fs: int = FS) -> np.ndarray:
    """Extract 1D signal from a single-lead image via dark-pixel centroid scan."""
    arr  = np.array(pil_img)
    H, W = arr.shape[:2]
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Black signal pixels: all channels < 100
    sig_mask = (R < 100) & (G < 100) & (B < 100)
    sig_out  = np.zeros(W, dtype=np.float32)

    for xc in range(W):
        rows = np.where(sig_mask[:, xc])[0]
        if len(rows) > 0:
            sig_out[xc] = 1.0 - (rows.mean() / H)   # top=high, bottom=low
        else:
            sig_out[xc] = np.nan

    # Interpolate NaNs
    nans = np.isnan(sig_out)
    if nans.any():
        xp = np.where(~nans)[0]
        if len(xp) > 1:
            sig_out = np.interp(np.arange(W), xp, sig_out[~nans])
        else:
            sig_out = np.full(W, 0.5)

    # Smooth, resample, bandpass, znorm
    sig = median_filter(sig_out.astype(np.float32), size=5)
    sig = resample(sig, seq_len).astype(np.float32)
    sig = bandpass(sig, fs=fs)
    mu  = sig.mean(); std = sig.std()
    if std > 1e-8:
        sig = (sig - mu) / std
    return sig


def digitize_uploaded_ecg(pil_image: Image.Image,
                            n_leads: int = 7,
                            seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Digitize a standard clinical 12-lead ECG printout image.

    Layout (calibrated from pixel analysis of real ECG scans):
        Left column  (col 0): I, II, III, aVR, aVL, aVF   (rows 0-5)
        Right column (col 1): V1, V2, V3, V4, V5, V6      (rows 0-5)

    Image is normalised to 2000x1413 internally for consistent thresholds.
    Signal pixels isolated as black (R,G,B < 80), separating from red grid
    and white background. Column-by-column median centroid scan extracts
    1D signal from each lead region.

    Returns float32 array [n_leads, seq_len].
    """
    pil_image = pil_image.convert('RGB')
    # Normalise to reference resolution for consistent pixel thresholds
    W, H = 2000, 1413
    pil_image = pil_image.resize((W, H), Image.LANCZOS)
    arr = np.array(pil_image)

    R = arr[:,:,0].astype(np.float32)
    G = arr[:,:,1].astype(np.float32)
    B = arr[:,:,2].astype(np.float32)

    # Black = signal  |  Red = grid  |  White = background
    sig_mask = (R < 80) & (G < 80) & (B < 80)

    # Layout constants (calibrated via pixel analysis of real ECG scans)
    # Header contains patient info + measurements (~20% of height)
    # Footer contains technical info bar (~3% of height)
    header_end   = 285    # first body row
    footer_start = 1373   # last body row
    body_H       = footer_start - header_end   # 1088 px
    row_h        = body_H / 6                  # 181 px per lead row
    v_inset      = int(row_h * 0.10)           # vertical margin per strip

    # Vertical dotted divider between left and right columns at col 1010
    div_col  = 1010
    cal_skip = 130   # skip calibration square + lead label text
    r_margin = 20    # skip right edge of each column

    col_bounds = [
        (cal_skip,           div_col - r_margin),   # left col: I,II,III,aVR,aVL,aVF
        (div_col + cal_skip, W       - r_margin),   # right col: V1,V2,V3,V4,V5,V6
    ]

    # Explicit grid position (row, col) for each of the 7 input leads
    lead_grid = [
        (0, 0),  # I
        (1, 0),  # II
        (2, 0),  # III
        (3, 0),  # aVR
        (4, 0),  # aVL
        (5, 0),  # aVF
        (0, 1),  # V1
    ]

    signals = []
    for lead_idx in range(min(n_leads, len(lead_grid))):
        grow, gcol = lead_grid[lead_idx]

        r0 = header_end + int(grow       * row_h) + v_inset
        r1 = header_end + int((grow + 1) * row_h) - v_inset
        c0, c1 = col_bounds[gcol]

        if r1 <= r0 or c1 <= c0:
            signals.append(np.zeros(seq_len, dtype=np.float32))
            continue

        region = sig_mask[r0:r1, c0:c1]
        rH, rW = region.shape

        # Column-by-column median centroid scan
        sig_out = np.full(rW, np.nan, dtype=np.float32)
        for xc in range(rW):
            dark = np.where(region[:, xc])[0]
            if len(dark) > 0:
                sig_out[xc] = 1.0 - (np.median(dark) / rH)

        # Detect Lead Off: fewer than 5% of columns have signal pixels
        valid_frac = np.sum(~np.isnan(sig_out)) / rW
        if valid_frac < 0.05:
            signals.append(np.zeros(seq_len, dtype=np.float32))
            continue

        # Interpolate NaN gaps
        nans = np.isnan(sig_out)
        if nans.any():
            xp = np.where(~nans)[0]
            sig_out = np.interp(np.arange(rW), xp, sig_out[~nans]).astype(np.float32)

        # Light median filter (size=3) to remove isolated noise without blurring QRS
        sig = median_filter(sig_out, size=3).astype(np.float32)

        # Resample to seq_len, bandpass 0.5-40 Hz, z-score
        sig = resample(sig, seq_len).astype(np.float32)
        sig = bandpass(sig, fs=FS)
        mu  = sig.mean(); std = sig.std()
        if std > 1e-8:
            sig = (sig - mu) / std
        else:
            sig = np.zeros(seq_len, dtype=np.float32)
        signals.append(sig)

    return np.stack(signals, axis=0)   # [n_leads, seq_len]


# ── Manual-crop helpers ───────────────────────────────────────────────────────

def default_lead_crops(img_w: int, img_h: int) -> dict:
    """
    Return default bounding boxes for the 7 input leads in image pixel coords.
    Boxes are derived from the same layout constants used in digitize_uploaded_ecg,
    but expressed relative to the *display* image size (img_w × img_h).

    Returns dict: lead_name → (x0, y0, x1, y1)  in display-image pixels.
    """
    # Reference resolution used internally
    REF_W, REF_H = 2000, 1413
    sx = img_w / REF_W
    sy = img_h / REF_H

    header_end   = 285
    footer_start = 1373
    body_H       = footer_start - header_end
    row_h        = body_H / 6
    v_inset      = int(row_h * 0.10)
    div_col      = 1010
    cal_skip     = 130
    r_margin     = 20

    col_bounds_ref = [
        (cal_skip,           div_col - r_margin),
        (div_col + cal_skip, REF_W   - r_margin),
    ]

    lead_grid = [
        (0, 0),  # I
        (1, 0),  # II
        (2, 0),  # III
        (3, 0),  # aVR
        (4, 0),  # aVL
        (5, 0),  # aVF
        (0, 1),  # V1
    ]

    boxes = {}
    for lead_name, (grow, gcol) in zip(INPUT_LEADS, lead_grid):
        r0 = header_end + int(grow       * row_h) + v_inset
        r1 = header_end + int((grow + 1) * row_h) - v_inset
        c0, c1 = col_bounds_ref[gcol]
        # Scale to display resolution
        boxes[lead_name] = (
            int(c0 * sx), int(r0 * sy),
            int(c1 * sx), int(r1 * sy),
        )
    return boxes


def render_crop_overlay(pil_img: Image.Image, boxes: dict,
                         active_lead: str = None) -> Image.Image:
    """
    Draw crop-box overlays on pil_img.
    active_lead box is drawn in bright orange; others in cyan.
    Returns a new PIL Image.
    """
    COLORS = {
        'I':   '#00BFFF', 'II':  '#00BFFF', 'III': '#00BFFF',
        'aVR': '#00BFFF', 'aVL': '#00BFFF', 'aVF': '#00BFFF',
        'V1':  '#00BFFF',
    }
    W, H = pil_img.size
    dpi  = 96
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(np.array(pil_img))
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)

    for name, (x0, y0, x1, y1) in boxes.items():
        color = '#FF8C00' if name == active_lead else COLORS.get(name, '#00BFFF')
        lw    = 2.5 if name == active_lead else 1.5
        rect  = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               linewidth=lw, edgecolor=color,
                               facecolor=color, alpha=0.08)
        ax.add_patch(rect)
        border = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(border)
        ax.text(x0 + 4, y0 + 14, name,
                color=color, fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.55, pad=1, linewidth=0))

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data-URL string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def roi_canvas_html(img_b64: str,
                    canvas_w: int,
                    canvas_h: int,
                    confirmed_boxes: dict,   # lead→(x0,y0,x1,y1) in canvas coords
                    active_lead: str,
                    input_key: str) -> str:
    """
    Return a self-contained HTML string that renders a drag-to-draw ROI canvas.
    The user draws a rectangle; on mouse-up the coords are written as JSON into
    a hidden <input id='roi_out'> and posted to Streamlit via
    window.parent.postMessage so st.query_params can pick it up.

    We instead write into a visible <textarea> that the user can copy —
    but the cleaner path is: embed the result in a <div> that Streamlit's
    components.v1.html() can read via its return value hack.

    Actually the simplest zero-dep approach: write coords into a named
    query-param via window.parent.postMessage with type 'streamlit:setComponentValue',
    which is the official Streamlit custom-component protocol.
    """
    # Serialise confirmed boxes for JS
    confirmed_js = json.dumps({
        ln: list(b) for ln, b in confirmed_boxes.items()
    })

    colors = {
        'I':'#6366f1','II':'#8b5cf6','III':'#a855f7',
        'aVR':'#06b6d4','aVL':'#0ea5e9','aVF':'#3b82f6',
        'V1':'#10b981',
    }
    active_color = colors.get(active_lead, '#f97316') if active_lead else '#f97316'

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; }}
  #wrap {{
    position: relative;
    width: {canvas_w}px;
    height: {canvas_h}px;
    cursor: crosshair;
    user-select: none;
  }}
  #ecg-img {{
    position: absolute; top:0; left:0;
    width: {canvas_w}px; height: {canvas_h}px;
    pointer-events: none;
  }}
  #overlay {{
    position: absolute; top:0; left:0;
    width: {canvas_w}px; height: {canvas_h}px;
  }}
  #hint {{
    margin-top: 6px;
    font-family: sans-serif; font-size: 13px;
    color: #374151;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 6px 10px;
  }}
  #coords-out {{
    margin-top: 6px;
    font-family: monospace; font-size: 12px;
    width: 100%; padding: 4px 8px;
    border: 1px solid #d1d5db; border-radius: 4px;
    background: #f3f4f6; color: #111827;
  }}
</style>
</head>
<body>
<div id="wrap">
  <img id="ecg-img" src="{img_b64}" draggable="false">
  <canvas id="overlay" width="{canvas_w}" height="{canvas_h}"></canvas>
</div>
<div id="hint">
  {"🖱️ Click and drag to draw a box around <b>" + active_lead + "</b>" if active_lead else "✅ All leads defined."}
</div>
<input id="coords-out" type="text" readonly
       placeholder="Drawn box coordinates will appear here…">

<script>
const canvas  = document.getElementById('overlay');
const ctx     = canvas.getContext('2d');
const coordEl = document.getElementById('coords-out');
const W = {canvas_w}, H = {canvas_h};

// ── Confirmed boxes (already drawn leads) ────────────────────────────
const confirmed = {confirmed_js};
const leadColors = {json.dumps(colors)};

function drawConfirmed() {{
  for (const [ln, b] of Object.entries(confirmed)) {{
    const [x0,y0,x1,y1] = b;
    const col = leadColors[ln] || '#22c55e';
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = col;
    ctx.fillRect(x0, y0, x1-x0, y1-y0);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(x0, y0, x1-x0, y1-y0);
    // label
    ctx.fillStyle = col;
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText('✓ ' + ln, x0+4, y0+14);
    ctx.restore();
  }}
}}

// ── Live drag-draw ───────────────────────────────────────────────────
let dragging = false, sx = 0, sy = 0, ex = 0, ey = 0;
const activeColor = '{active_color}';
const activeLead  = {json.dumps(active_lead)};

function redraw() {{
  ctx.clearRect(0, 0, W, H);
  drawConfirmed();
  if (dragging || (ex !== 0 && ey !== 0)) {{
    const rx = Math.min(sx,ex), ry = Math.min(sy,ey);
    const rw = Math.abs(ex-sx),  rh = Math.abs(ey-sy);
    ctx.save();
    ctx.globalAlpha = 0.15;
    ctx.fillStyle = activeColor;
    ctx.fillRect(rx, ry, rw, rh);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = activeColor;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6,3]);
    ctx.strokeRect(rx, ry, rw, rh);
    // corner handles
    const hs = 6;
    ctx.setLineDash([]);
    [[rx,ry],[rx+rw,ry],[rx,ry+rh],[rx+rw,ry+rh]].forEach(([cx,cy]) => {{
      ctx.fillStyle = activeColor;
      ctx.fillRect(cx-hs/2, cy-hs/2, hs, hs);
    }});
    // dimension label
    if (rw > 30 && rh > 15) {{
      const label = activeLead ? `${{activeLead}}: ${{rw}}×${{rh}}px` : `${{rw}}×${{rh}}px`;
      ctx.font = 'bold 12px sans-serif';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillRect(rx+4, ry+4, tw+8, 18);
      ctx.fillStyle = activeColor;
      ctx.fillText(label, rx+8, ry+17);
    }}
    ctx.restore();
  }}
}}

function getPos(e) {{
  const r = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return [
    Math.max(0, Math.min(W, clientX - r.left)),
    Math.max(0, Math.min(H, clientY - r.top)),
  ];
}}

if (activeLead) {{
  canvas.addEventListener('mousedown',  e => {{ [sx,sy]=getPos(e); ex=sx; ey=sy; dragging=true; }});
  canvas.addEventListener('mousemove',  e => {{ if (!dragging) return; [ex,ey]=getPos(e); redraw(); }});
  canvas.addEventListener('mouseup',    e => {{
    dragging=false; [ex,ey]=getPos(e); redraw();
    const x0=Math.round(Math.min(sx,ex)), y0=Math.round(Math.min(sy,ey));
    const x1=Math.round(Math.max(sx,ex)), y1=Math.round(Math.max(sy,ey));
    const val = JSON.stringify([x0,y0,x1,y1]);
    coordEl.value = val;
    // Send to Streamlit component host
    window.parent.postMessage({{type:'streamlit:setComponentValue', value:val}}, '*');
  }});
  canvas.addEventListener('mouseleave', e => {{ if (dragging) {{ dragging=false; redraw(); }} }});
  // Touch support
  canvas.addEventListener('touchstart', e=>{{ e.preventDefault(); [sx,sy]=getPos(e); ex=sx; ey=sy; dragging=true; }}, {{passive:false}});
  canvas.addEventListener('touchmove',  e=>{{ e.preventDefault(); if(!dragging) return; [ex,ey]=getPos(e); redraw(); }}, {{passive:false}});
  canvas.addEventListener('touchend',   e=>{{ e.preventDefault(); dragging=false; redraw();
    const x0=Math.round(Math.min(sx,ex)), y0=Math.round(Math.min(sy,ey));
    const x1=Math.round(Math.max(sx,ex)), y1=Math.round(Math.max(sy,ey));
    coordEl.value = JSON.stringify([x0,y0,x1,y1]);
    window.parent.postMessage({{type:'streamlit:setComponentValue', value:coordEl.value}}, '*');
  }}, {{passive:false}});
}}

// Initial draw
redraw();
</script>
</body>
</html>
"""


def digitize_from_crops(pil_image: Image.Image,
                          boxes: dict,
                          seq_len: int = SEQ_LEN,
                          fs: int = FS) -> np.ndarray:
    """
    Digitize each lead from its manually defined bounding box.
    boxes: dict  lead_name → (x0, y0, x1, y1) in display-image pixels.
    Returns float32 [7, seq_len].
    """
    arr = np.array(pil_image.convert('RGB'))
    signals = []

    for lead_name in INPUT_LEADS:
        x0, y0, x1, y1 = boxes[lead_name]
        # Clamp to image bounds
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(arr.shape[1], x1); y1 = min(arr.shape[0], y1)

        if x1 <= x0 or y1 <= y0:
            signals.append(np.zeros(seq_len, dtype=np.float32))
            continue

        region = arr[y0:y1, x0:x1]
        R, G, B = region[:,:,0], region[:,:,1], region[:,:,2]
        sig_mask = (R.astype(np.float32) < 80) & \
                   (G.astype(np.float32) < 80) & \
                   (B.astype(np.float32) < 80)

        rH, rW = sig_mask.shape
        sig_out = np.full(rW, np.nan, dtype=np.float32)

        for xc in range(rW):
            dark = np.where(sig_mask[:, xc])[0]
            if len(dark) > 0:
                sig_out[xc] = 1.0 - (np.median(dark) / rH)

        valid_frac = np.sum(~np.isnan(sig_out)) / rW
        if valid_frac < 0.05:
            signals.append(np.zeros(seq_len, dtype=np.float32))
            continue

        nans = np.isnan(sig_out)
        if nans.any():
            xp = np.where(~nans)[0]
            if len(xp) > 1:
                sig_out = np.interp(np.arange(rW), xp, sig_out[~nans]).astype(np.float32)
            else:
                sig_out = np.full(rW, 0.5, dtype=np.float32)

        sig = median_filter(sig_out, size=3).astype(np.float32)
        sig = resample(sig, seq_len).astype(np.float32)
        sig = bandpass(sig, fs=fs)
        mu  = sig.mean(); std = sig.std()
        sig = (sig - mu) / std if std > 1e-8 else np.zeros(seq_len, dtype=np.float32)
        signals.append(sig)

    return np.stack(signals, axis=0)


# ── Model classes


# ── Model classes (copied from notebook) ─────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, cond_d: int, ch: int):
        super().__init__()
        self.scale = nn.Linear(cond_d, ch)
        self.shift = nn.Linear(cond_d, ch)

    def forward(self, x, c):
        return x * (1 + self.scale(c).unsqueeze(-1)) + self.shift(c).unsqueeze(-1)


class ResBlk(nn.Module):
    def __init__(self, ci, co, cd=None, drop=0.1):
        super().__init__()
        g = lambda ch: min(8, ch)
        self.body = nn.Sequential(
            nn.GroupNorm(g(ci), ci), nn.GELU(),
            nn.Conv1d(ci, co, 3, padding=1), nn.Dropout(drop),
            nn.GroupNorm(g(co), co), nn.GELU(),
            nn.Conv1d(co, co, 3, padding=1),
        )
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()
        self.film = FiLM(cd, co) if cd else None

    def forward(self, x, c=None):
        h = self.body(x)
        if self.film and c is not None:
            h = self.film(h, c)
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.p = nn.Conv1d(ch, ch, 4, 2, 1)
    def forward(self, x): return self.p(x)


class Up(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.u = nn.ConvTranspose1d(ci, co, 4, 2, 1)
    def forward(self, x, skip):
        x = self.u(x)
        d = skip.shape[-1] - x.shape[-1]
        if d > 0: x = F.pad(x, [0, d])
        return torch.cat([x[:, :, :skip.shape[-1]], skip], dim=1)


class LeadGenerator(nn.Module):
    def __init__(self, ni=7, no=5, ch=48, cd=1024, drop=0.1):
        super().__init__()
        self.cproj = nn.Sequential(
            nn.Linear(cd, ch*4), nn.GELU(), nn.Linear(ch*4, ch*4))
        C = ch * 4
        self.e1, self.d1 = ResBlk(ni,   ch,   C, drop), Down(ch)
        self.e2, self.d2 = ResBlk(ch,   ch*2, C, drop), Down(ch*2)
        self.e3, self.d3 = ResBlk(ch*2, ch*4, C, drop), Down(ch*4)
        self.e4, self.d4 = ResBlk(ch*4, ch*8, C, drop), Down(ch*8)
        self.m1 = ResBlk(ch*8, ch*8, C, drop)
        self.m2 = ResBlk(ch*8, ch*8, C, drop)
        self.u4, self.r4 = Up(ch*8, ch*8),  ResBlk(ch*16, ch*8, C, drop)
        self.u3, self.r3 = Up(ch*8, ch*4),  ResBlk(ch*8,  ch*4, C, drop)
        self.u2, self.r2 = Up(ch*4, ch*2),  ResBlk(ch*4,  ch*2, C, drop)
        self.u1, self.r1 = Up(ch*2, ch),    ResBlk(ch*2,  ch,   C, drop)
        self.out = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), nn.GELU(), nn.Conv1d(ch, no, 1))

    def forward(self, x, clip_emb):
        c  = self.cproj(clip_emb)
        s1 = self.e1(x,  c); x = self.d1(s1)
        s2 = self.e2(x,  c); x = self.d2(s2)
        s3 = self.e3(x,  c); x = self.d3(s3)
        s4 = self.e4(x,  c); x = self.d4(s4)
        x  = self.m2(self.m1(x, c), c)
        x  = self.r4(self.u4(x, s4), c)
        x  = self.r3(self.u3(x, s3), c)
        x  = self.r2(self.u2(x, s2), c)
        x  = self.r1(self.u1(x, s1), c)
        return self.out(x)


# ── CLIP Extractor ────────────────────────────────────────────────────────────
class CLIPExtractor:
    def __init__(self, model_id='openai/clip-vit-large-patch14', device='cpu'):
        from transformers import CLIPProcessor, CLIPModel
        self.proc = CLIPProcessor.from_pretrained(model_id)
        clip      = CLIPModel.from_pretrained(model_id)
        self.enc  = clip.vision_model.to(device)
        self.dim  = clip.vision_model.config.hidden_size   # 1024 for ViT-L
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()
        self.device = device

    @torch.no_grad()
    def extract(self, images: list) -> np.ndarray:
        inp = self.proc(images=images, return_tensors='pt', padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        return self.enc(**inp).pooler_output.cpu().numpy()


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LeadGenerator model weights…")
def load_model(weights_path: str, cdim: int = 1024,
               base_ch: int = 48, device: str = 'cpu') -> LeadGenerator:
    model = LeadGenerator(ni=7, no=5, ch=base_ch, cd=cdim)
    ckpt  = torch.load(weights_path, map_location=device)
    # Accept both plain state_dict and wrapped checkpoint
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading CLIP ViT-L/14 (first run only)…")
def load_clip(device: str = 'cpu') -> CLIPExtractor:
    return CLIPExtractor(device=device)


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model: LeadGenerator,
                  clip:  CLIPExtractor,
                  signals_7: np.ndarray,        # [7, 2500]
                  device: str = 'cpu') -> np.ndarray:
    """
    Given 7 input leads, return predicted 5 leads [5, 2500].
    Steps:
      1. Render full 7-lead (padded to 12) as red-grid image
      2. Extract CLIP embedding
      3. Run LeadGenerator
    """
    # Build 12-lead array for CLIP: use zeros for missing leads
    sig12 = np.zeros((12, SEQ_LEN), dtype=np.float32)
    sig12[:7] = signals_7
    clip_img = render_redgrid(sig12, ALL_LEADS, img_sz=(448, 448))

    # CLIP embedding
    feats = clip.extract([clip_img])                    # [1, 1024]
    x     = torch.FloatTensor(signals_7).unsqueeze(0)   # [1, 7, 2500]
    c     = torch.FloatTensor(feats)                    # [1, 1024]

    pred  = model(x.to(device), c.to(device))          # [1, 5, 2500]
    return pred.cpu().numpy()[0]                        # [5, 2500]


# ── Visualisation helpers ─────────────────────────────────────────────────────
def plot_comparison(signals_7: np.ndarray, pred_5: np.ndarray) -> plt.Figure:
    """Plot predicted V2–V6 as interactive matplotlib figure."""
    full_12  = np.concatenate([signals_7, pred_5], axis=0)
    L        = min(full_12.shape[1], 1250)               # show 2.5 s
    t        = np.linspace(0, L / FS, L)
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Reconstructed Leads V2–V6', fontweight='bold', fontsize=13)
    colors = ['#2ECC71', '#E74C3C', '#3498DB', '#9B59B6', '#F39C12']
    for i, (name, color) in enumerate(zip(TARGET_LEADS, colors)):
        axes[i].plot(t, pred_5[i, :L], color=color, lw=1.2, label=name)
        axes[i].set_ylabel(name, rotation=0, labelpad=28, fontsize=10, fontweight='bold')
        axes[i].grid(alpha=0.3); axes[i].set_ylim(-3, 3)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    return fig


def pil_to_bytes(img: Image.Image, fmt='PNG') -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    # ── Header ───────────────────────────────────────────────────────
    st.title("🫀 ECG Lead Generator · 7 → 12 Leads")
    st.markdown(
        "Upload an ECG image or signal file. "
        "The model reconstructs **V2, V3, V4, V5, V6** from "
        "**I, II, III, aVR, aVL, aVF, V1**."
    )
    st.divider()

    # ── Sidebar — model loading ───────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Model Settings")

        st.markdown("**Step 1 — Upload model weights**")
        st.caption(
            "Export from Colab using: "
            "`torch.save(model.state_dict(), 'lead_generator_weights.pt')`"
        )
        weights_file = st.file_uploader(
            "lead_generator_weights.pt",
            type=['pt', 'pth'],
            key='weights'
        )

        cdim    = st.number_input("CLIP dim", value=1024, step=256,
                                   help="1024 for ViT-L/14 (default), 768 for ViT-B/32")
        base_ch = st.number_input("BASE_CH", value=48, step=16,
                                   help="Must match the value used during training")
        device  = 'cpu'   # Streamlit Cloud has no GPU

        model_ready = False
        clip_ready  = False

        if weights_file is not None:
            # Save to temp file so torch.load can read it
            tmp_path = '/tmp/lead_generator_weights.pt'
            with open(tmp_path, 'wb') as f:
                f.write(weights_file.read())
            try:
                model = load_model(tmp_path, cdim=cdim, base_ch=base_ch, device=device)
                st.success(f"✓ LeadGenerator loaded  ({sum(p.numel() for p in model.parameters()):,} params)")
                model_ready = True
            except Exception as e:
                st.error(f"Failed to load model: {e}")

        st.divider()
        st.markdown("**Step 2 — Load CLIP**")
        if st.button("Load CLIP ViT-L/14", disabled=not model_ready):
            with st.spinner("Downloading CLIP (~1.7 GB, first run only)…"):
                try:
                    clip = load_clip(device=device)
                    st.session_state['clip'] = clip
                    st.success(f"✓ CLIP ready  (dim={clip.dim})")
                except Exception as e:
                    st.error(f"CLIP load failed: {e}")

        clip_ready = 'clip' in st.session_state

        st.divider()
        st.markdown(
            "**Input format for signal files:**\n"
            "- `.npy` — numpy array shape `[7, 2500]` or `[12, 2500]`\n"
            "- `.csv` — 7 rows × 2500 columns (one row per lead)\n\n"
            "Lead order: I, II, III, aVR, aVL, aVF, V1"
        )

    # ── Main panel — input ────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Input")
        input_mode = st.radio(
            "Input mode",
            ["ECG Image (PNG/JPG)", "Signal File (.npy / .csv)"],
            horizontal=True
        )

        signals_7 = None

        if input_mode == "ECG Image (PNG/JPG)":
            uploaded_img = st.file_uploader(
                "Upload ECG image (standard 12-lead layout: 6 rows × 2 columns)",
                type=['png', 'jpg', 'jpeg']
            )
            if uploaded_img:
                # Cache image in session state so sliders don't re-upload
                img_key = uploaded_img.name + str(uploaded_img.size)
                if st.session_state.get('_img_key') != img_key:
                    pil_img = Image.open(uploaded_img).convert('RGB')
                    st.session_state['_pil_img']  = pil_img
                    st.session_state['_img_key']  = img_key
                    # Initialise default crop boxes
                    W_disp, H_disp = pil_img.size
                    st.session_state['_crop_boxes'] = default_lead_crops(W_disp, H_disp)
                    st.session_state['_signals_7']  = None

                pil_img    = st.session_state['_pil_img']
                W_disp, H_disp = pil_img.size

                # ── Crop mode toggle ──────────────────────────────────────
                crop_mode = st.toggle(
                    "✂️ Manual ROI crop mode",
                    value=st.session_state.get('_crop_mode', False),
                    help="Draw a rectangle directly on the ECG image to define each lead's region",
                )
                st.session_state['_crop_mode'] = crop_mode

                if crop_mode:
                    boxes      = st.session_state['_crop_boxes']
                    lead_idx   = st.session_state.get('_roi_lead_idx', 0)
                    leads_done = st.session_state.get('_roi_leads_done', set())

                    # ── Progress chips ────────────────────────────────────
                    chip_html = ""
                    for i, ln in enumerate(INPUT_LEADS):
                        if ln in leads_done:
                            chip_html += (
                                f'<span style="background:#22c55e;color:#fff;padding:3px 10px;'
                                f'border-radius:999px;margin:2px;font-size:13px;font-weight:600">'
                                f'✓ {ln}</span>'
                            )
                        elif i == lead_idx:
                            chip_html += (
                                f'<span style="background:#f97316;color:#fff;padding:3px 10px;'
                                f'border-radius:999px;margin:2px;font-size:13px;font-weight:700;'
                                f'box-shadow:0 0 0 3px #fed7aa">● {ln}</span>'
                            )
                        else:
                            chip_html += (
                                f'<span style="background:#e5e7eb;color:#6b7280;padding:3px 10px;'
                                f'border-radius:999px;margin:2px;font-size:13px">○ {ln}</span>'
                            )
                    st.markdown(
                        f'<div style="margin:6px 0 12px">{chip_html}</div>',
                        unsafe_allow_html=True,
                    )

                    active_lead = INPUT_LEADS[lead_idx] if lead_idx < len(INPUT_LEADS) else None

                    if active_lead:
                        st.markdown(
                            f"**🖱️ Click and drag on the image to draw a box around "
                            f"lead `{active_lead}`, then click Confirm.**"
                        )
                    else:
                        st.success("✅ All leads defined! Click **Apply ROI crops & digitize** below.")

                    # ── Scale image to canvas width ───────────────────────
                    CANVAS_W = 860
                    scale    = CANVAS_W / W_disp
                    CANVAS_H = int(H_disp * scale)

                    # Scale confirmed boxes to canvas coordinates
                    confirmed_canvas = {
                        ln: (
                            int(boxes[ln][0] * scale), int(boxes[ln][1] * scale),
                            int(boxes[ln][2] * scale), int(boxes[ln][3] * scale),
                        )
                        for ln in leads_done
                    }

                    # Embed the ECG image as a base64 data-URL (no external file needed)
                    img_resized = pil_img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
                    img_b64     = pil_to_b64(img_resized)

                    # ── Render the HTML5 canvas ROI tool ─────────────────
                    # The component writes coords into a hidden text_input via
                    # postMessage; we read it back through a st.text_input below.
                    html_src = roi_canvas_html(
                        img_b64        = img_b64,
                        canvas_w       = CANVAS_W,
                        canvas_h       = CANVAS_H,
                        confirmed_boxes= confirmed_canvas,
                        active_lead    = active_lead,
                        input_key      = f"roi_{active_lead}",
                    )
                    import streamlit.components.v1 as components
                    roi_value = components.html(
                        html_src,
                        height=CANVAS_H + 80,
                        scrolling=False,
                    )

                    # ── Coord readback: hidden text_input synced by JS ────
                    # postMessage sets component value; we also show a manual
                    # paste field as reliable fallback (works in all deployments)
                    st.caption(
                        "After drawing, the coordinates appear below automatically. "
                        "If not, right-click the box coordinates in the canvas and paste them here:"
                    )
                    raw_coords = st.text_input(
                        "Box coordinates (auto-filled or paste JSON [x0,y0,x1,y1])",
                        value=st.session_state.get(f'_raw_{active_lead}', ''),
                        key=f'coord_input_{active_lead}',
                        placeholder='[120, 45, 480, 110]',
                        label_visibility='collapsed',
                    )
                    st.session_state[f'_raw_{active_lead}'] = raw_coords

                    # Parse and show confirm / skip buttons
                    parsed_box = None
                    if raw_coords.strip():
                        try:
                            vals = json.loads(raw_coords.strip())
                            if (isinstance(vals, list) and len(vals) == 4 and
                                    all(isinstance(v, (int, float)) for v in vals)):
                                # Convert canvas coords → original image coords
                                cx0 = int(vals[0] / scale); cy0 = int(vals[1] / scale)
                                cx1 = int(vals[2] / scale); cy1 = int(vals[3] / scale)
                                if cx1 > cx0 + 4 and cy1 > cy0 + 4:
                                    parsed_box = (cx0, cy0, cx1, cy1)
                                    st.caption(
                                        f"📐 Box: x={cx0}–{cx1}, y={cy0}–{cy1} "
                                        f"({cx1-cx0}×{cy1-cy0} px in original image)"
                                    )
                        except (json.JSONDecodeError, ValueError):
                            st.warning("Could not parse coordinates — draw again or check the JSON.")

                    if active_lead:
                        confirm_col, skip_col = st.columns([3, 1])
                        with confirm_col:
                            confirm_disabled = parsed_box is None
                            if st.button(
                                f"✔ Confirm ROI for **{active_lead}**",
                                type="primary",
                                disabled=confirm_disabled,
                                use_container_width=True,
                            ):
                                boxes[active_lead] = parsed_box
                                st.session_state['_crop_boxes']            = boxes
                                st.session_state[f'_raw_{active_lead}']   = ''
                                leads_done.add(active_lead)
                                st.session_state['_roi_leads_done']        = leads_done
                                st.session_state['_roi_lead_idx']          = lead_idx + 1
                                st.rerun()
                        with skip_col:
                            if st.button("⏭ Skip", use_container_width=True,
                                         help="Keep the auto-detected region for this lead"):
                                st.session_state[f'_raw_{active_lead}'] = ''
                                leads_done.add(active_lead)
                                st.session_state['_roi_leads_done'] = leads_done
                                st.session_state['_roi_lead_idx']   = lead_idx + 1
                                st.rerun()

                        # Live preview of current region
                        preview_box = parsed_box if parsed_box else boxes[active_lead]
                        st.markdown(f"**Preview — `{active_lead}` current region:**")
                        st.image(pil_img.crop(preview_box), use_container_width=True)

                    else:
                        # All done — summary overlay
                        summary_scale = CANVAS_W / W_disp
                        summary_img = render_crop_overlay(
                            pil_img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS),
                            {ln: (int(boxes[ln][0]*summary_scale), int(boxes[ln][1]*summary_scale),
                                  int(boxes[ln][2]*summary_scale), int(boxes[ln][3]*summary_scale))
                             for ln in boxes},
                        )
                        st.image(summary_img, caption="Final ROI layout — all 7 leads", use_container_width=True)

                    # ── Action buttons ────────────────────────────────────
                    st.divider()
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("↩ Reset all ROIs to auto-detected"):
                            st.session_state['_crop_boxes']     = default_lead_crops(W_disp, H_disp)
                            st.session_state['_roi_lead_idx']   = 0
                            st.session_state['_roi_leads_done'] = set()
                            st.session_state['_signals_7']      = None
                            for ln in INPUT_LEADS:
                                st.session_state.pop(f'_raw_{ln}', None)
                            st.rerun()
                    with btn_col2:
                        if st.button(
                            "✅ Apply ROI crops & digitize",
                            type='primary',
                            disabled=(len(leads_done) == 0),
                            use_container_width=True,
                        ):
                            with st.spinner("Digitizing from ROI crops…"):
                                try:
                                    signals_7 = digitize_from_crops(
                                        pil_img,
                                        st.session_state['_crop_boxes'],
                                        seq_len=SEQ_LEN,
                                    )
                                    st.session_state['_signals_7'] = signals_7
                                    st.success(
                                        f"✓ Digitized from {len(leads_done)} manual ROI(s): "
                                        f"shape {signals_7.shape}, "
                                        f"range [{signals_7.min():.2f}, {signals_7.max():.2f}]"
                                    )
                                except Exception as e:
                                    st.error(f"Digitization failed: {e}")

                    signals_7 = st.session_state.get('_signals_7')

                else:
                    # Auto mode — show image and auto-digitize
                    st.image(pil_img, caption="Uploaded ECG", use_container_width=True)

                    if st.session_state.get('_signals_7') is None:
                        with st.spinner("Auto-digitizing ECG image…"):
                            try:
                                signals_7 = digitize_uploaded_ecg(pil_img, n_leads=7, seq_len=SEQ_LEN)
                                st.session_state['_signals_7'] = signals_7
                                st.success(
                                    f"✓ Auto-digitized: shape {signals_7.shape}, "
                                    f"range [{signals_7.min():.2f}, {signals_7.max():.2f}]"
                                )
                            except Exception as e:
                                st.error(f"Digitization failed: {e}")
                    else:
                        signals_7 = st.session_state['_signals_7']
                        st.success(
                            f"✓ Digitized: shape {signals_7.shape}, "
                            f"range [{signals_7.min():.2f}, {signals_7.max():.2f}]"
                        )

        else:  # Signal file
            uploaded_sig = st.file_uploader(
                "Upload signal file",
                type=['npy', 'csv']
            )
            if uploaded_sig:
                try:
                    if uploaded_sig.name.endswith('.npy'):
                        arr = np.load(io.BytesIO(uploaded_sig.read()))
                    else:
                        import pandas as pd
                        arr = pd.read_csv(uploaded_sig, header=None).values.astype(np.float32)

                    # Accept [7, L], [12, L], or [L, 7], [L, 12]
                    if arr.shape[0] in (7, 12) and arr.ndim == 2:
                        arr = arr[:7]
                    elif arr.shape[1] in (7, 12) and arr.ndim == 2:
                        arr = arr[:, :7].T
                    else:
                        st.error(f"Unexpected shape {arr.shape}. Expected [7,2500] or [12,2500].")
                        arr = None

                    if arr is not None:
                        # Resample to SEQ_LEN if needed
                        if arr.shape[1] != SEQ_LEN:
                            arr = np.stack([
                                resample(arr[i], SEQ_LEN).astype(np.float32)
                                for i in range(7)
                            ])
                        signals_7 = preprocess_signals(arr)
                        st.success(f"✓ Loaded: shape {signals_7.shape},"
                                   f" range [{signals_7.min():.2f}, {signals_7.max():.2f}]")
                except Exception as e:
                    st.error(f"Failed to load signal file: {e}")

        # Preview input leads
        if signals_7 is not None:
            st.markdown("**Input leads preview (first 2.5 s):**")
            L_prev = min(signals_7.shape[1], 1250)
            t_prev = np.linspace(0, L_prev / FS, L_prev)
            fig_in, axes_in = plt.subplots(7, 1, figsize=(12, 8), sharex=True)
            for i, name in enumerate(INPUT_LEADS):
                axes_in[i].plot(t_prev, signals_7[i, :L_prev], 'k-', lw=0.8)
                axes_in[i].set_ylabel(name, rotation=0, labelpad=24, fontsize=8)
                axes_in[i].set_ylim(-3, 3); axes_in[i].grid(alpha=0.25)
                axes_in[i].tick_params(labelsize=6)
            axes_in[-1].set_xlabel('Time (s)', fontsize=8)
            fig_in.suptitle('7 Input Leads', fontweight='bold', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_in); plt.close(fig_in)

    # ── Main panel — output ───────────────────────────────────────────
    with col2:
        st.subheader("📊 Reconstruction")

        run_btn = st.button(
            "▶  Reconstruct V2–V6",
            disabled=(signals_7 is None or not model_ready or not clip_ready),
            type='primary',
            use_container_width=True
        )

        if not model_ready:
            st.info("Upload model weights in the sidebar to enable reconstruction.")
        elif not clip_ready:
            st.info("Click **Load CLIP** in the sidebar.")
        elif signals_7 is None:
            st.info("Upload an ECG image or signal file on the left.")

        if run_btn and signals_7 is not None and model_ready and clip_ready:
            with st.spinner("Running inference…"):
                try:
                    clip  = st.session_state['clip']
                    pred5 = run_inference(model, clip, signals_7, device=device)

                    st.success("✓ Reconstruction complete")

                    # Waveform comparison
                    fig_pred = plot_comparison(signals_7, pred5)
                    st.pyplot(fig_pred); plt.close(fig_pred)

                    # Full 12-lead clinical report
                    st.markdown("**Full 12-lead Clinical ECG Report:**")
                    full_12 = np.concatenate([signals_7, pred5], axis=0)
                    report  = render_redgrid(full_12, ALL_LEADS, img_sz=(896, 672))
                    st.image(report, use_container_width=True)

                    # Downloads
                    st.markdown("**Download results:**")
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            "⬇️  Clinical ECG (PNG)",
                            data=pil_to_bytes(report),
                            file_name="ecg_12lead_report.png",
                            mime="image/png"
                        )
                    with dl2:
                        buf_npy = io.BytesIO()
                        np.save(buf_npy, pred5)
                        st.download_button(
                            "⬇️  Predicted signals (.npy)",
                            data=buf_npy.getvalue(),
                            file_name="predicted_V2_V6.npy",
                            mime="application/octet-stream"
                        )

                    # Per-lead metrics (self-check: RMSE of pred vs flat baseline)
                    st.markdown("**Predicted lead statistics:**")
                    stat_cols = st.columns(5)
                    for i, (name, col) in enumerate(zip(TARGET_LEADS, stat_cols)):
                        with col:
                            st.metric(
                                label=name,
                                value=f"{pred5[i].std():.3f}",
                                delta=f"peak {pred5[i].max():.2f}",
                                help="Std dev of reconstructed lead amplitude (z-scored)"
                            )

                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.exception(e)

    # ── Footer ────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "**Model:** CLIP-ViT-L/14 conditioned 1D U-Net  ·  "
        "**Dataset:** PTB-XL (PhysioNet)  ·  "
        "**Input:** 7 leads → **Output:** V2–V6  ·  "
        "For research and testing only — not for clinical diagnosis."
    )


if __name__ == '__main__':
    main()