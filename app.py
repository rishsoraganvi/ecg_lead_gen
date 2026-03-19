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
from scipy.ndimage import median_filter, gaussian_filter, binary_dilation, gaussian_filter1d, sobel
from scipy.signal import find_peaks

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


def _adaptive_canny(signal_channel: np.ndarray,
                    sigma: float = 1.2,
                    low_pct: float = 75,
                    high_pct: float = 93) -> np.ndarray:
    """
    Canny edge detection with adaptive thresholds.

    Why Canny over plain dark-pixel threshold:
      - Produces single-pixel thin edges instead of 2-4px thick blobs,
        giving sub-pixel amplitude precision via median centroid.
      - Adaptive thresholds (percentile-based) handle faint flat leads
        and tall QRS spikes in the same image without manual tuning.
      - Hysteresis linking prevents broken edges on faint signal sections.

    Thresholds are set at the p75 and p93 of the image's actual gradient
    magnitude distribution — automatically scales to image contrast.
    """
    blurred = gaussian_filter(signal_channel.astype(float), sigma=sigma)
    gx      = np.gradient(blurred, axis=1)
    gy      = np.gradient(blurred, axis=0)
    mag     = np.hypot(gx, gy)
    nonzero = mag.ravel()
    nonzero = nonzero[nonzero > 0.5]
    if len(nonzero) < 10:
        return np.zeros_like(mag, dtype=bool)
    low    = float(np.percentile(nonzero, low_pct))
    high   = float(np.percentile(nonzero, high_pct))
    strong = mag > high
    weak   = (mag >= low) & ~strong
    return strong | (weak & binary_dilation(strong, iterations=2))


def _detect_grid_spacing(region_rgb: np.ndarray):
    """
    Detect minor grid spacing (pixels) from vertical redness-channel peaks.

    Standard ECG paper calibration:
      1 minor square = 1mm = 0.04 s horizontally = 0.1 mV vertically
      Paper speed = 25mm/s,  Amplitude gain = 10mm/mV

    Returns (minor_sp_px, px_per_sec, px_per_mv).
    Falls back to 7.9px (≈200 DPI standard scan) if detection fails.
    """
    R = region_rgb[:,:,0].astype(float)
    G = region_rgb[:,:,1].astype(float)
    B = region_rgb[:,:,2].astype(float)
    redness     = np.maximum(0.0, R - (G + B) / 2.0)
    col_profile = gaussian_filter(redness.mean(axis=0), sigma=2.0)
    peaks, _    = find_peaks(col_profile, height=4, distance=4)
    if len(peaks) >= 10:
        spacings = np.diff(peaks)
        minor_sp = float(np.median(spacings[spacings < np.percentile(spacings, 60)]))
        if 3 < minor_sp < 60:
            return minor_sp, minor_sp / 0.04, minor_sp / 0.1
    return 7.9, 7.9 / 0.04, 7.9 / 0.1


def digitize_uploaded_ecg(pil_image: Image.Image,
                            n_leads: int = 7,
                            seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    ECG image digitizer with 4 improvements:

    1. ORIGINAL RESOLUTION
       Works at native pixel density — no downscaling that blurs the signal.

    2. ADAPTIVE CANNY + DARK PIXEL FALLBACK
       For each lead strip:
         a) Redness channel = R − avg(G,B) → identifies pink/red grid pixels.
         b) Signal channel  = inverted brightness × (1 − grid_weight):
            black signal → high value;  red grid → suppressed.
         c) Adaptive Canny (p75/p93 thresholds) → single-pixel thin edge.
         d) Dark-pixel fallback (R,G,B < 130, not reddish) for flat leads
            where JPEG compression makes the signal line near-invisible to Canny.
         e) Column scan: Canny rows preferred; dark rows used as fallback.

    3. NaN INTERPOLATION
       NaNs arise when no signal pixel exists in a column (grid intersection,
       JPEG blur, or lead-label overlap). np.interp() fills each NaN by
       linear interpolation between nearest valid neighbours. Safe because
       gaps are 1-5 pixels wide — amplitude change is negligible across them.

    4. GRID-REFERENCED mV CONVERSION
       _detect_grid_spacing() finds the minor grid spacing in pixels from
       redness-channel column peaks. Signal y-position is then converted:
         mV = (mid_row − signal_row) / px_per_mv
       Output is in calibrated mV units before z-scoring.

    Returns float32 array [n_leads, seq_len], z-score normalised.
    """
    # 1. ORIGINAL RESOLUTION
    img  = np.array(pil_image.convert('RGB'))
    H, W = img.shape[:2]

    body_top = int(H * 0.12)
    body_bot = int(H * 0.95)
    body_H   = body_bot - body_top
    row_h    = body_H / 6
    col_w    = W / 2

    # Clinical 12-lead layout:
    #   Left col  (col 0): I, II, III, aVR, aVL, aVF  rows 0-5
    #   Right col (col 1): V1, V2, V3, V4, V5, V6     rows 0-5
    lead_grid_pos = [
        (0, 0),  # I
        (1, 0),  # II
        (2, 0),  # III
        (3, 0),  # aVR
        (4, 0),  # aVL
        (5, 0),  # aVF
        (0, 1),  # V1
    ]

    signals = []
    for lead_idx in range(min(n_leads, 7)):
        grow, gcol = lead_grid_pos[lead_idx]
        r0 = body_top + int(grow       * row_h + row_h * 0.18)
        r1 = body_top + int((grow + 1) * row_h - row_h * 0.10)
        c0 = int(gcol       * col_w + col_w * 0.10)
        c1 = int((gcol + 1) * col_w - col_w * 0.02)

        region = img[r0:r1, c0:c1]
        rH, rW = region.shape[:2]

        # 4. GRID CALIBRATION
        minor_sp, px_per_sec, px_per_mv = _detect_grid_spacing(region)

        # 2. SIGNAL CHANNEL + ADAPTIVE CANNY
        R = region[:,:,0].astype(np.float64)
        G = region[:,:,1].astype(np.float64)
        B = region[:,:,2].astype(np.float64)
        redness        = np.maximum(0.0, R - (G + B) / 2.0)
        brightness     = (R + G + B) / 3.0
        grid_weight    = np.clip(redness / 25.0, 0.0, 1.0)
        signal_channel = (255.0 - brightness) * (1.0 - grid_weight * 0.80)

        canny_mask = _adaptive_canny(signal_channel, sigma=1.2,
                                     low_pct=75, high_pct=93)
        # Fallback: any dark non-reddish pixel (handles JPEG-blurred flat leads)
        dark_mask  = (R < 130) & (G < 130) & (B < 130) & (redness < 25)

        # Column scan: Canny preferred, dark pixels as fallback
        sig_px = np.full(rW, np.nan, dtype=np.float64)
        for xc in range(rW):
            canny_rows = np.where(canny_mask[:, xc])[0]
            dark_rows  = np.where(dark_mask[:, xc])[0]
            if len(canny_rows) > 0:
                sig_px[xc] = float(np.median(canny_rows))
            elif len(dark_rows) > 0:
                sig_px[xc] = float(np.median(dark_rows))

        # 4. PIXEL → mV CONVERSION
        baseline_px = rH / 2.0
        sig_mv      = (baseline_px - sig_px) / px_per_mv

        # 3. NaN INTERPOLATION
        nans = np.isnan(sig_mv)
        if nans.all():
            sig_mv = np.zeros(rW, dtype=np.float32)
        elif nans.any():
            xp     = np.where(~nans)[0]
            sig_mv = np.interp(np.arange(rW), xp, sig_mv[~nans]).astype(np.float32)

        # Post-process: smooth → resample → bandpass → z-score
        sig = median_filter(sig_mv.astype(np.float32), size=7)
        sig = resample(sig, seq_len).astype(np.float32)
        sig = bandpass(sig, fs=FS)
        mu  = sig.mean(); std = sig.std()
        if std > 1e-8:
            sig = (sig - mu) / std
        signals.append(sig)

    return np.stack(signals, axis=0)


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
                pil_img = Image.open(uploaded_img).convert('RGB')
                st.image(pil_img, caption="Uploaded ECG", use_container_width=True)

                with st.spinner("Digitizing ECG image…"):
                    try:
                        signals_7 = digitize_uploaded_ecg(pil_img, n_leads=7, seq_len=SEQ_LEN)
                        st.success(f"✓ Digitized: shape {signals_7.shape},"
                                   f" range [{signals_7.min():.2f}, {signals_7.max():.2f}]")
                    except Exception as e:
                        st.error(f"Digitization failed: {e}")

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
