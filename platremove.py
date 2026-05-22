import os
import uuid
import base64
import io
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import cv2

try:
    from flask_cors import CORS
    _has_cors = True
except ImportError:
    _has_cors = False

try:
    from rembg import remove as rembg_remove
    _has_rembg = True
except ImportError:
    _has_rembg = False

app = Flask(__name__)
if _has_cors:
    CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///autolens.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER']    = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

db = SQLAlchemy(app)

class CarImage(db.Model):
    id             = db.Column(db.String(50),  primary_key=True)
    filename       = db.Column(db.String(255))
    original_path  = db.Column(db.String(500))
    processed_path = db.Column(db.String(500))
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
    os.makedirs(app.config['UPLOAD_FOLDER'],    exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)


# ═══════════════════════════════════════════════════════════
#  DETECTION ENGINE  —  5 independent methods, best-score wins
# ═══════════════════════════════════════════════════════════

def _score_candidate(x, y, w, h, img_w, img_h):
    """
    Score a candidate box on how plate-like it is.
    Returns 0..100. Anything >= 45 is worth keeping.

    KEY RULE: Number plates are ALWAYS in the bottom 35% of the car image.
    Anything above that (windows, doors, body) must be rejected hard.
    """
    if w <= 0 or h <= 0:
        return 0

    aspect      = w / h
    area_ratio  = (w * h) / (img_w * img_h)
    cx_ratio    = (x + w / 2) / img_w   # horizontal center position
    cy_ratio    = (y + h / 2) / img_h   # vertical center position

    score = 0

    # ── HARD REJECT: plate cannot be in top 55% of image ─────
    # Window / door / body areas are above 55% — never a plate
    if cy_ratio < 0.55:
        return 0

    # ── HARD REJECT: aspect ratio must be wide rectangle ─────
    # Indian plates: ~4.7:1. Anything taller than wide = not a plate
    if aspect < 1.8 or aspect > 7.0:
        return 0

    # ── HARD REJECT: too large (entire bumper/door) ───────────
    if w > img_w * 0.55 or h > img_h * 0.15:
        return 0

    # ── HARD REJECT: too small ────────────────────────────────
    if w < max(60, img_w * 0.06) or h < max(14, img_h * 0.018):
        return 0

    # ── Aspect ratio scoring ──────────────────────────────────
    # Tight band first — Indian plates ~3.5:1 to 5.5:1
    if   3.0 <= aspect <= 5.5:  score += 40
    elif 2.5 <= aspect <= 6.0:  score += 22
    elif 1.8 <= aspect <= 7.0:  score += 8

    # ── Area scoring ─────────────────────────────────────────
    if   0.008 <= area_ratio <= 0.07: score += 25
    elif 0.004 <= area_ratio <= 0.12: score += 10

    # ── Absolute size scoring ────────────────────────────────
    min_w = max(60, img_w * 0.06)
    max_w = img_w * 0.55
    min_h = max(14, img_h * 0.018)
    max_h = img_h * 0.12
    if min_w <= w <= max_w and min_h <= h <= max_h:
        score += 20

    # ── Vertical position: MUST be in bottom 45% ─────────────
    if   0.70 <= cy_ratio <= 0.95: score += 15   # ideal: very bottom
    elif 0.55 <= cy_ratio <= 0.70: score += 5    # acceptable: lower-mid

    # ── Horizontal center ────────────────────────────────────
    if   0.20 <= cx_ratio <= 0.80: score += 8
    elif 0.10 <= cx_ratio <= 0.90: score += 3

    return score


def _method_edge(img_cv, img_w, img_h):
    """Canny edge → contours → scored candidates."""
    candidates = []
    gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur    = cv2.bilateralFilter(gray, 9, 17, 17)
    edges   = cv2.Canny(blur, 25, 180)

    # Also try with different sigma
    edges2  = cv2.Canny(blur, 50, 250)
    combined = cv2.bitwise_or(edges, edges2)

    for e in [combined, edges]:
        cnts, _ = cv2.findContours(e, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area descending, take more candidates
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:40]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            s = _score_candidate(x, y, w, h, img_w, img_h)
            if s >= 45:
                candidates.append((s, x, y, w, h, 'edge'))

    return candidates


def _method_morph(img_cv, img_w, img_h):
    """Morphological operations to isolate plate-like rectangles."""
    candidates = []
    gray  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Blackhat highlights dark text on light background
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat  = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kern)

    # CLAHE for contrast enhancement
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blackhat)

    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close gaps horizontally
    close_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed     = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kern)
    closed     = cv2.dilate(closed, close_kern, iterations=1)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        s = _score_candidate(x, y, w, h, img_w, img_h)
        if s >= 45:
            candidates.append((s, x, y, w, h, 'morph'))

    return candidates


def _method_color(img_cv, img_w, img_h):
    """HSV color detection for white/yellow/red plates and badges."""
    candidates = []
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    masks = []

    # White plate
    masks.append(cv2.inRange(hsv, np.array([0,  0,  170]), np.array([180, 45, 255])))
    # Yellow plate
    masks.append(cv2.inRange(hsv, np.array([18, 60, 100]), np.array([38, 255, 255])))
    # Spinny red badge (hue 0-10 and 160-180)
    r1 = cv2.inRange(hsv, np.array([0,  130, 70]),  np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([158,130, 70]),  np.array([180,255, 255]))
    masks.append(cv2.bitwise_or(r1, r2))
    # Cars24 orange
    masks.append(cv2.inRange(hsv, np.array([8, 150, 80]),  np.array([22, 255, 255])))

    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 6))

    for mask in masks:
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kern, iterations=2)
        m = cv2.dilate(m, kern, iterations=1)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            s = _score_candidate(x, y, w, h, img_w, img_h)
            if s >= 40:
                # Expand to capture surrounding text / badge area
                px = int(w * 0.35); py = int(h * 0.55)
                nx = max(0, x-px);  ny = max(0, y-py)
                nw = min(img_w-nx, w+px*2); nh = min(img_h-ny, h+py*2)
                s2 = _score_candidate(nx, ny, nw, nh, img_w, img_h)
                candidates.append((max(s, s2), nx, ny, nw, nh, 'color'))

    return candidates


def _method_sobel(img_cv, img_w, img_h):
    """Sobel gradient to find high horizontal-frequency regions (text)."""
    candidates = []
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelX = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    absX   = cv2.convertScaleAbs(sobelX)
    _, thr = cv2.threshold(absX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    thr  = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kern, iterations=2)
    thr  = cv2.dilate(thr, kern, iterations=1)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        s = _score_candidate(x, y, w, h, img_w, img_h)
        if s >= 45:
            candidates.append((s, x, y, w, h, 'sobel'))

    return candidates


def _method_white_rect(img_cv, img_w, img_h):
    """Find bright white rectangular regions (dealer plates are white-background)."""
    candidates = []
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Focus on lower portion only
    roi_y  = int(img_h * 0.30)
    roi    = gray[roi_y:, :]

    _, thr = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    thr    = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kern, iterations=2)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        y += roi_y   # adjust back to full image coords

        fill = cv2.countNonZero(thr[y - roi_y:y - roi_y + h, x:x + w]) / max(w * h, 1)
        if fill < 0.35:
            continue

        # Skip if it's too wide (likely a bumper/hood)
        if w > img_w * 0.60:
            continue

        s = _score_candidate(x, y, w, h, img_w, img_h)
        if s >= 42:
            # Check center-ish horizontally
            cx = (x + w / 2) / img_w
            if 0.15 <= cx <= 0.85:
                candidates.append((s, x, y, w, h, 'white_rect'))

    return candidates


def detect_number_plate(image_path):
    """
    Run all 5 methods, pick highest-scoring candidate.
    Returns (x, y, w, h) or None.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[detect] Cannot read image: {image_path}")
            return None

        img_h, img_w = img.shape[:2]
        print(f"\n🔍 Detecting plate in {img_w}×{img_h} image...")

        all_candidates = []
        all_candidates += _method_edge(img,       img_w, img_h)
        all_candidates += _method_morph(img,      img_w, img_h)
        all_candidates += _method_color(img,      img_w, img_h)
        all_candidates += _method_sobel(img,      img_w, img_h)
        all_candidates += _method_white_rect(img, img_w, img_h)

        if not all_candidates:
            print("[detect] No candidates found")
            return None

        # Sort by score descending
        all_candidates.sort(key=lambda c: c[0], reverse=True)

        # Print top 5 for debug
        for i, (s, x, y, w, h, method) in enumerate(all_candidates[:5]):
            print(f"  #{i+1} score={s:3d} [{method:10s}] ({x},{y},{w},{h}) ar={w/max(h,1):.2f}")

        best = all_candidates[0]
        s, x, y, w, h, method = best

        # Clamp to image bounds
        x = max(0, x); y = max(0, y)
        w = min(img_w - x, w); h = min(img_h - y, h)

        print(f"\n✅ Best plate: ({x},{y},{w},{h}) score={s} via [{method}]")
        return (x, y, w, h)

    except Exception as e:
        print(f"[detect] Error: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════
#  REMOVAL ENGINE
# ═══════════════════════════════════════════════════════════

def _load_font(size):
    paths = [
        "arial.ttf", "Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, int(size))
        except Exception:
            pass
    return ImageFont.load_default()


def apply_removal(image_path, output_path, x, y, w, h, mode='caryanams'):
    """
    Apply chosen removal mode to the given box on the image.
    Saves result to output_path.
    """
    try:
        img  = Image.open(image_path).convert('RGB')
        iw, ih = img.size

        # Clamp
        x = max(0, x); y = max(0, y)
        w = min(iw - x, w); h = min(ih - y, h)

        if mode == 'blur':
            region = img.crop((x, y, x+w, y+h))
            # Pixelate: downscale then upscale
            block  = max(8, h // 4)
            small  = region.resize((max(1, w//block), max(1, h//block)), Image.NEAREST)
            pix    = small.resize((w, h), Image.NEAREST)
            pix    = pix.filter(ImageFilter.GaussianBlur(radius=2))
            img.paste(pix, (x, y))

        elif mode == 'white':
            draw = ImageDraw.Draw(img)
            draw.rectangle([x, y, x+w, y+h], fill=(255, 255, 255))

        elif mode == 'black':
            draw = ImageDraw.Draw(img)
            draw.rectangle([x, y, x+w, y+h], fill=(0, 0, 0))

        elif mode == 'caryanams':
            # ── Transparent logo cutout — no white box, no border ──
            # 1. Erase plate with surrounding car-body color
            # 2. Paste only "Caryanams" text + golden icon (transparent PNG) on top
            logo_pasted = False
            logo_paths = [
                'caryanams_logo_clean.png',
                os.path.join(os.path.dirname(__file__), 'caryanams_logo_clean.png'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'caryanams_logo_clean.png'),
                'static/caryanams_logo_clean.png',
                os.path.join('static', 'caryanams_logo_clean.png'),
            ]

            for logo_path in logo_paths:
                if os.path.exists(logo_path):
                    try:
                        logo_raw = Image.open(logo_path).convert('RGBA')
                        arr_l    = np.array(logo_raw)

                        # Step 1: Remove white background → transparent
                        white_mask        = (arr_l[:,:,0]>240) & (arr_l[:,:,1]>240) & (arr_l[:,:,2]>240)
                        arr_l[:,:,3]      = np.where(white_mask, 0, 255)

                        # Step 2: Tight-crop to actual logo content only
                        content_mask = arr_l[:,:,3] > 10
                        rows_m = np.any(content_mask, axis=1)
                        cols_m = np.any(content_mask, axis=0)
                        rmin, rmax = np.where(rows_m)[0][[0, -1]]
                        cmin, cmax = np.where(cols_m)[0][[0, -1]]
                        logo_t = Image.fromarray(arr_l).crop(
                            (cmin, rmin, cmax + 1, rmax + 1)
                        )
                        print(f"[apply] Transparent logo cropped to {logo_t.size}")

                        # Step 3: Fill plate area with WHITE background
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
                        print(f"[apply] Plate filled with white")

                        # Step 4: Scale logo to fit plate box
                        margin   = max(3, h // 10)
                        target_w = w - margin * 2
                        target_h = h - margin * 2
                        lw, lh   = logo_t.size
                        scale_f  = min(target_w / lw, target_h / lh)
                        nw = max(1, int(lw * scale_f))
                        nh = max(1, int(lh * scale_f))
                        logo_s   = logo_t.resize((nw, nh), Image.LANCZOS)

                        # Step 5: Paste transparent logo centered on erased plate area
                        base_rgba = img.convert('RGBA')
                        px = x + (w - nw) // 2
                        py = y + (h - nh) // 2
                        base_rgba.paste(logo_s, (px, py), logo_s)
                        img = base_rgba.convert('RGB')

                        logo_pasted = True
                        print(f"[apply] Logo cutout pasted at ({px},{py}) size {nw}×{nh}")
                        break
                    except Exception as le:
                        print(f"[apply] Logo paste failed ({logo_path}): {le}")
                        continue

            if not logo_pasted:
                # Fallback: white box with text if logo file missing
                print("[apply] Logo file not found, using text fallback")
                draw = ImageDraw.Draw(img)
                draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
                f_main = _load_font(max(9, h * 0.38))
                tmp_d  = ImageDraw.Draw(Image.new("RGB", (1, 1)))
                try:
                    bb = tmp_d.textbbox((0, 0), "Caryanams", font=f_main)
                    tw, th = bb[2]-bb[0], bb[3]-bb[1]
                except Exception:
                    tw = w // 2; th = h // 3
                draw.text((x + (w - tw) // 2, y + (h - th) // 2),
                          "Caryanams", fill=(17, 56, 110), font=f_main)

        img.save(output_path, 'JPEG', quality=95)
        print(f"[apply] Saved → {output_path}")
        return True

    except Exception as e:
        print(f"[apply] Error: {e}")
        traceback.print_exc()
        return False


def detect_and_remove(image_path, output_path, mode='caryanams'):
    """Full pipeline: detect plate then remove it."""
    plate = detect_number_plate(image_path)
    if plate is None:
        return False, None
    x, y, w, h = plate
    ok = apply_removal(image_path, output_path, x, y, w, h, mode)
    return ok, plate


# ═══════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('plate_remover.html')


@app.route('/api/upload-car', methods=['POST'])
def upload_car():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Empty filename'}), 400

        uid = str(uuid.uuid4())[:8]
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            ext = '.jpg'

        fname         = f"car_{uid}{ext}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(original_path)

        car = CarImage(id=uid, filename=fname, original_path=original_path)
        db.session.add(car)
        db.session.commit()

        return jsonify({
            'success': True,
            'id': uid,
            'original_url': '/' + original_path.replace('\\', '/')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-plate/<image_id>', methods=['GET'])
def detect_plate_api(image_id):
    try:
        car = CarImage.query.get_or_404(image_id)
        plate = detect_number_plate(car.original_path)

        if plate:
            x, y, w, h = plate
            iw, ih = Image.open(car.original_path).size
            return jsonify({
                'detected': True,
                'x': x, 'y': y, 'width': w, 'height': h,
                'img_width': iw, 'img_height': ih,
                'message': f'Plate at ({x},{y}) size {w}×{h}'
            })
        else:
            iw, ih = Image.open(car.original_path).size
            return jsonify({
                'detected': False,
                'img_width': iw, 'img_height': ih,
                'message': 'Auto-detection failed. Use manual selection.'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ═══════════════════════════════════════════════════════════
#  COMBINED: Plate Remove + BG Remove + White Oval BG
# ═══════════════════════════════════════════════════════════

SHOWROOM_BG_PATHS = [
    'showroom_bg.jpeg',
    'showroom_bg.jpg',
    'showroom_bg.png',
    'static/showroom_bg.jpeg',
    'static/showroom_bg.jpg',
    'static/showroom_bg.png',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'showroom_bg.jpeg'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'showroom_bg.jpg'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'showroom_bg.png'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'showroom_bg.jpeg'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'showroom_bg.jpg'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'showroom_bg.png'),
]

def _load_showroom_bg():
    """Load the fixed showroom background image."""
    for p in SHOWROOM_BG_PATHS:
        if os.path.exists(p):
            print(f"[bg] Loaded showroom background: {p}")
            return Image.open(p).convert('RGBA')
    print("[bg] Showroom BG not found — using white fallback")
    return None


def apply_showroom_background(car_rgba, car_size_pct=75):
    """
    Composite transparent car onto the fixed showroom floor background.
    car_size_pct: 30–100  →  percentage of canvas width the car should fill
    """
    bg_orig = _load_showroom_bg()

    # ── Canvas setup ──────────────────────────────────────────
    if bg_orig:
        bg_w, bg_h = bg_orig.size
        # Pehle white RGB base banao, phir BG uske upar flatten karo
        # Isse koi bhi transparent area white hoga, black nahi
        base   = Image.new('RGB', (bg_w, bg_h), (255, 255, 255))
        bg_rgb = bg_orig.convert('RGBA')
        base.paste(bg_rgb, mask=bg_rgb.split()[3])
        canvas = base.convert('RGBA')
    else:
        bg_w, bg_h = 1382, 752
        canvas = Image.new('RGBA', (bg_w, bg_h), (255, 255, 255, 255))

    # ── Floor line ────────────────────────────────────────────
    floor_y = int(bg_h * 0.95)

    # ── Scale car based on slider value ──────────────────────
    size_scale   = max(0.30, min(1.00, car_size_pct / 100.0))
    car_w, car_h = car_rgba.size
    target_car_w = int(bg_w * size_scale)
    scale        = target_car_w / car_w
    target_car_h = int(car_h * scale)
    print(f"[bg] car_size_pct={car_size_pct}% target_w={target_car_w}px")

    # Car height floor se upar fit honi chahiye (car_y >= 0)
    max_car_h = floor_y - 5
    if target_car_h > max_car_h:
        scale        = max_car_h / car_h
        target_car_w = int(car_w * scale)
        target_car_h = max_car_h

    # Car width canvas se zyada nahi
    if target_car_w > bg_w:
        scale        = bg_w / car_w
        target_car_w = bg_w
        target_car_h = int(car_h * scale)

    car_scaled = car_rgba.resize((target_car_w, target_car_h), Image.LANCZOS)
    print(f"[bg] Car scaled to {target_car_w}x{target_car_h} on {bg_w}x{bg_h} canvas")

    # ── Position ──────────────────────────────────────────────
        # ── Position (CENTERED SHOWROOM LOOK) ─────────────────────
    car_x = max(0, (bg_w - target_car_w) // 2)

    center_offset = int(bg_h * 0.08)

    car_y = max(
        0,
        (bg_h - target_car_h) // 2 + center_offset
    )

    # Safety: floor ke niche na jaye
    max_y = floor_y - target_car_h + 25

    if car_y > max_y:
        car_y = max_y

    # ── Contact shadow on floor ───────────────────────────────
    shadow_layer = Image.new('RGBA', (bg_w, bg_h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_layer)

    sh_cx = bg_w // 2
    sh_cy = floor_y - int(target_car_h * 0.01)
    sh_rx = int(target_car_w * 0.42)
    sh_ry = int(target_car_h * 0.045)

    shadow_layers = [
        (sh_rx + 50, sh_ry + 18, 12),
        (sh_rx + 30, sh_ry + 11, 22),
        (sh_rx + 14, sh_ry + 6,  35),
        (sh_rx,      sh_ry,      50),
        (sh_rx - 16, sh_ry - 3,  65),
        (sh_rx - 32, sh_ry - 6,  50),
        (sh_rx - 46, sh_ry - 9,  35),
    ]

    for rx, ry, alpha in shadow_layers:
        if rx <= 0 or ry <= 0:
            continue

        sdraw.ellipse(
            [sh_cx-rx, sh_cy-ry, sh_cx+rx, sh_cy+ry],
            fill=(30, 30, 35, alpha)
        )

    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=14))
    canvas = Image.alpha_composite(canvas, shadow_layer)

    # ── Paste car onto showroom floor ─────────────────────────
    canvas.paste(car_scaled, (car_x, car_y), car_scaled)

    # ── Convert to RGB — no black areas ─────────────────────
    final = Image.new('RGB', (bg_w, bg_h), (255, 255, 255))
    final.paste(canvas.convert('RGB'), mask=canvas.split()[3])

    return final


def process_all_in_one(image_path, output_path, mode='caryanams', manual=None, car_size_pct=60):
    """
    ONE-CLICK pipeline:
      Step 1 → Detect & remove number plate
      Step 2 → Remove background (rembg)
      Step 3 → Place car on white oval background
    Returns (ok, plate_info)
    """
    try:
        print(f"\n{'='*55}")
        print(f"  ONE-CLICK PIPELINE START")
        print(f"{'='*55}")

        # ── STEP 1: Number Plate Remove ──────────────────────
        temp_plate_path = output_path.replace('.png', '_step1.jpg')

        plate_info = None
        if manual:
            x = int(manual['x']); y = int(manual['y'])
            w = int(manual['w']); h = int(manual['h'])
            ok1 = apply_removal(image_path, temp_plate_path, x, y, w, h, mode)
            plate_info = {'x': x, 'y': y, 'width': w, 'height': h}
            print(f"[step1] Manual plate remove: ok={ok1}")
        else:
            plate = detect_number_plate(image_path)
            if plate:
                x, y, w, h = plate
                ok1 = apply_removal(image_path, temp_plate_path, x, y, w, h, mode)
                plate_info = {'x': x, 'y': y, 'width': w, 'height': h}
                print(f"[step1] Auto plate remove: ok={ok1} at ({x},{y},{w},{h})")
            else:
                # No plate found — still continue with BG remove
                import shutil
                shutil.copy(image_path, temp_plate_path)
                ok1 = True
                print("[step1] No plate detected — skipping plate step, continuing BG remove")

        source_for_bg = temp_plate_path if os.path.exists(temp_plate_path) else image_path

        # ── STEP 2: Background Remove ─────────────────────────
        if not _has_rembg:
            print("[step2] rembg not installed — skipping BG remove")
            # Just do white oval without BG remove
            img_pil = Image.open(source_for_bg).convert('RGBA')
        else:
            print("[step2] Removing background with rembg...")
            with open(source_for_bg, 'rb') as f:
                raw = f.read()
            removed = rembg_remove(raw)
            img_pil = Image.open(io.BytesIO(removed)).convert('RGBA')
            print(f"[step2] BG removed, size={img_pil.size}")

            # ── WHEEL PROTECTION ──────────────────────────────
            # rembg sometimes cuts off wheels/tyres at the bottom.
            # Fix: for the bottom 35% of the car, restore ONLY very dark pixels
            # (tyres/rims are near-black). Gray showroom floors (r,g,b > 80) are
            # intentionally left transparent so the floor is fully removed.
            try:
                orig_pil = Image.open(source_for_bg).convert('RGBA')
                orig_arr = np.array(orig_pil)
                rmbg_arr = np.array(img_pil)

                iw_r, ih_r = img_pil.size
                # Bottom 35% of image = wheel zone
                wheel_zone_top = int(ih_r * 0.65)

                wheel_orig = orig_arr[wheel_zone_top:, :, :]
                wheel_rmbg = rmbg_arr[wheel_zone_top:, :, :]

                orig_r = wheel_orig[:, :, 0].astype(int)
                orig_g = wheel_orig[:, :, 1].astype(int)
                orig_b = wheel_orig[:, :, 2].astype(int)

                # ── TYRE PIXEL DETECTION ────────────────────────────
                # Tyres / rims are VERY dark (near-black). Threshold set at 80
                # so gray showroom floors (r~150-200) are NOT restored — only
                # true rubber/rim pixels (r < 80) are brought back.
                # Also check: not too uniform gray (floor tends to be r≈g≈b).
                is_very_dark = (orig_r < 80) & (orig_g < 80) & (orig_b < 80)

                # Additional: allow slightly lighter rim pixels (dark gray)
                # but only if there's significant darkness difference from floor
                is_dark_rim  = (orig_r < 120) & (orig_g < 120) & (orig_b < 120)
                # For these, check the pixel is much darker than a typical floor:
                # floor pixels tend to be r≈g≈b≈150+, so if all channels < 120 it's likely rim
                is_tyre_pixel   = is_very_dark | is_dark_rim  # covers tyre + rim + arch

                wrongly_removed = (wheel_rmbg[:, :, 3] < 128) & is_tyre_pixel

                wheel_rmbg[wrongly_removed, 0] = wheel_orig[wrongly_removed, 0]
                wheel_rmbg[wrongly_removed, 1] = wheel_orig[wrongly_removed, 1]
                wheel_rmbg[wrongly_removed, 2] = wheel_orig[wrongly_removed, 2]
                wheel_rmbg[wrongly_removed, 3] = 255

                rmbg_arr[wheel_zone_top:, :, :] = wheel_rmbg
                img_pil = Image.fromarray(rmbg_arr.astype(np.uint8), 'RGBA')
                restored = int(wrongly_removed.sum())
                print(f"[step2] Wheel protection: restored {restored} tyre pixels in bottom 35%")
                print(f"[step2] Floor pixels intentionally kept transparent (threshold: RGB < 120)")
            except Exception as wp_err:
                print(f"[step2] Wheel protection skipped: {wp_err}")

        # ── STEP 3: Showroom Background ──────────────────────
        print("[step3] Compositing onto showroom floor background...")
        final_img = apply_showroom_background(img_pil, car_size_pct=car_size_pct)
        final_img.save(output_path, 'PNG', optimize=True)
        print(f"[step3] Saved final → {output_path}")

        # Cleanup temp
        try:
            if os.path.exists(temp_plate_path):
                os.remove(temp_plate_path)
        except Exception:
            pass

        print(f"{'='*55}\n  PIPELINE DONE ✅\n{'='*55}\n")
        return True, plate_info

    except Exception as e:
        print(f"[pipeline] ERROR: {e}")
        traceback.print_exc()
        return False, None


@app.route('/api/process-car/<image_id>', methods=['POST'])
def process_car_api(image_id):
    """
    ONE-CLICK endpoint: plate remove + BG remove + white oval BG.
    """
    try:
        car = CarImage.query.get(image_id)
        if not car:
            return jsonify({'success': False, 'message': 'Image not found'}), 404

        data          = request.get_json() or {}
        mode          = data.get('mode', 'caryanams')
        manual        = data.get('manual')
        car_size_pct  = int(data.get('car_size_pct', 60))   # 30–100

        output_path = os.path.join(
            app.config['PROCESSED_FOLDER'],
            f'final_{image_id}.png'
        )

        ok, plate_info = process_all_in_one(car.original_path, output_path, mode, manual, car_size_pct)

        if ok and os.path.exists(output_path):
            car.processed_path = output_path
            db.session.commit()

            try:
                with open(output_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
            except Exception:
                b64 = None

            msg = '✅ Plate removed + Background removed + White oval background applied!'
            if not _has_rembg:
                msg = '⚠️ Plate removed + White oval applied (rembg not installed for BG removal)'

            return jsonify({
                'success': True,
                'processed_url': '/' + output_path.replace('\\', '/'),
                'preview_b64': b64,
                'plate': plate_info,
                'message': msg
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Processing failed. Try manual plate selection.'
            })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/remove-plate/<image_id>', methods=['POST'])
def remove_plate_api(image_id):
    """Legacy route — redirects to combined pipeline."""
    return process_car_api(image_id)


@app.route('/api/download/<image_id>')
def download_image(image_id):
    car  = CarImage.query.get_or_404(image_id)
    path = car.processed_path or car.original_path
    fname = f'caryanams_{car.id[:8]}.png'
    # HD: re-save with max quality if PNG
    try:
        hd_path = path.replace('.png', '_hd.png')
        if not os.path.exists(hd_path):
            img = Image.open(path)
            img.save(hd_path, 'PNG', optimize=False, compress_level=1)
        return send_file(hd_path, as_attachment=True, download_name=fname)
    except Exception:
        return send_file(path, as_attachment=True, download_name=fname)


@app.route('/api/gallery', methods=['GET'])
def gallery_api():
    """Return all processed cars for gallery page."""
    try:
        cars = CarImage.query.filter(CarImage.processed_path != None).order_by(CarImage.created_at.desc()).all()
        result = []
        for car in cars:
            if car.processed_path and os.path.exists(car.processed_path):
                result.append({
                    'id': car.id,
                    'filename': car.filename,
                    'processed_url': '/' + car.processed_path.replace('\\', '/'),
                    'original_url':  '/' + car.original_path.replace('\\', '/'),
                    'created_at': car.created_at.strftime('%d %b %Y, %I:%M %p') if car.created_at else ''
                })
        return jsonify({'success': True, 'cars': result, 'total': len(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/gallery')
def gallery_page():
    """Gallery HTML page."""
    gallery_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gallery.html')
    if os.path.exists(gallery_path):
        return send_file(gallery_path)
    return "<h2>gallery.html not found — platremove.py ke saath same folder mein rakho</h2>", 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗  CARYANAMS — Number Plate Remover  v2")
    print("="*60)
    print("✅  Multi-method detection (5 algorithms)")
    print("✅  Manual selection fallback")
    print("✅  Works on all car images")
    print("\n🌐  Open: http://localhost:5055")
    print("="*60 + "\n")
    app.run(debug=True, port=5055)