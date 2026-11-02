"""SmartRupee analysis engine: preprocessing, region extraction, ORB, SSIM, MobileNetV2, verdict."""
import os, json, glob, base64, re
import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim_fn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(BASE, "data", "references")
MODEL_PATH = os.path.join(BASE, "models", "genuine_feature_extractor.keras")
ANOMALY_PATH = os.path.join(BASE, "models", "genuine_reference_anomaly.keras")
DISTRIBUTION_PATH = os.path.join(BASE, "models", "genuine_reference_distribution.npz")
EVAL_PATH = os.path.join(BASE, "reports", "evaluation.json")

W, H = 1200, 520  # canonical aligned-note size
# Heuristic thresholds. They are NOT calibrated claims - tune them with your own labelled data.
CFG = dict(ratio=0.75, inl_ok=20, inl_bad=8, ssim_ok=0.50, ssim_bad=0.30,
           min_texture=12.0, min_edge=0.01, max_glare=0.25)

STAGES = ["Image uploaded successfully", "Image preprocessing", "Banknote feature extraction",
          "Corner-wise analysis", "ORB feature matching", "SSIM similarity analysis",
          "Genuine-reference comparison", "Final reference report"]

# id, label, (x0, y0, x1, y1) as fractions of the aligned note, is_corner
REGIONS = [
    ("top_left", "Top-left corner", (0.00, 0.00, 0.24, 0.40), True),
    ("top_right", "Top-right corner", (0.76, 0.00, 1.00, 0.40), True),
    ("bottom_left", "Bottom-left corner", (0.00, 0.60, 0.24, 1.00), True),
    ("bottom_right", "Bottom-right corner", (0.76, 0.60, 1.00, 1.00), True),
    ("thread", "Security-thread band (visual only)", (0.46, 0.00, 0.56, 1.00), False),
    ("centre_left", "Central-left design region", (0.26, 0.15, 0.44, 0.85), False),
    ("centre_right", "Central-right design region", (0.58, 0.15, 0.74, 0.85), False),
]
FEATURE_REGIONS = [
    ("watermark", "Watermark window", (0.20, 0.12, 0.34, 0.38), False),
    ("security_thread", "Security thread", (0.45, 0.08, 0.50, 0.92), False),
    ("serial_number", "Serial number area", (0.76, 0.60, 0.98, 0.78), False),
    ("portrait", "Portrait area", (0.35, 0.18, 0.48, 0.70), False),
]
REGION_FEATURES = {
    "top_left": "Security printing and design details",
    "top_right": "Printed design details",
    "bottom_left": "Micro-lettering and surrounding design details",
    "bottom_right": "Serial number and surrounding printed details",
    "thread": "Security thread",
    "centre_left": "Watermark-related area, where applicable",
    "centre_right": "Portrait and surrounding design",
    "watermark": "Watermark visibility requires transmitted light",
    "security_thread": "Embedded security thread (visual inspection only)",
    "serial_number": "Serial number and surrounding printed details",
    "portrait": "Portrait and surrounding design",
}
COL = {"green": (80, 200, 80), "red": (60, 60, 235), "yellow": (0, 200, 240)}
PHYSICAL = ["Tactile (raised-print) features", "UV fluorescence", "Infrared response",
            "Watermark and see-through registration under transmitted light",
            "Colour-shift ink angle response", "Paper and thread feel"]
VISUAL_INDICATORS = [
    dict(name="Watermark inconsistency", assessment="Inspect against a verified reference under transmitted light; missing or unclear evidence is inconclusive."),
    dict(name="Security-thread inconsistency", assessment="Check position, continuity and appearance against a same-series reference."),
    dict(name="See-through register mismatch", assessment="Requires front and back alignment under transmitted light."),
    dict(name="Poor micro-lettering", assessment="May indicate printing irregularity, but image resolution or magnification can also cause an inconclusive result."),
    dict(name="Printing irregularities", assessment="Visible smudging, alignment or pattern inconsistency requires a suitable reference comparison."),
    dict(name="Serial-number irregularity", assessment="Compare format, spacing and print quality against verified references for the same series."),
]

OCR_READER = None
OCR_READER_ERROR = None


def regional_text_assessment(crop_image):
    """Return non-identifying OCR evidence for printed-text regions."""
    global OCR_READER, OCR_READER_ERROR
    try:
        if OCR_READER is None:
            import easyocr
            OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        results = OCR_READER.readtext(gray, detail=1, paragraph=False)
        if not results:
            return dict(status="No readable text detected", text_confidence=None)
        usable = [float(confidence) for _, _, confidence in results if float(confidence) >= 0.2]
        if not usable:
            return dict(status="Text detection unclear", text_confidence=None)
        return dict(
            status="Printed text detected",
            text_confidence=round(float(max(usable)), 3),
        )
    except Exception as exc:
        OCR_READER_ERROR = str(exc)
        return dict(status="OCR unavailable", text_confidence=None)


def b64(img, q=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def crop(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def order_pts(p):
    p = np.array(p, dtype="float32"); s = p.sum(1); d = np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], dtype="float32")


def quality(img):
    s = 800 / max(img.shape[:2]); g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s) if s < 1 else img, cv2.COLOR_BGR2GRAY)
    blur, br = float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())
    glare = float((g >= 250).mean())
    dark = float((g <= 35).mean())
    highlight_mask = np.uint8(g >= 250) * 255
    highlight_blocks = cv2.connectedComponentsWithStats(highlight_mask, connectivity=8)[0] - 1
    issues, score = [], 100
    if blur < 40: issues.append("Image is blurry"); score -= 35
    if br < 60: issues.append("Image is too dark"); score -= 25
    if br > 200 or glare > 0.12: issues.append("Glare or overexposure"); score -= 25
    if dark > 0.25: issues.append("Large dark areas reduce visible detail"); score -= 20
    if min(img.shape[:2]) < 400: issues.append("Low resolution"); score -= 25
    score = max(0, score)
    return dict(score=score, level="Good" if score >= 75 else "Fair" if score >= 50 else "Poor",
                blur=round(blur, 1), blur_status="Blurry" if blur < 40 else "Acceptable",
                brightness=round(br, 1), glare=round(glare, 3),
                dark_area=round(dark, 3), highlight_regions=int(highlight_blocks),
                lighting="Poor" if br < 60 or br > 200 or glare > 0.12 else "Good",
                issues=issues)


def detect_note(img):
    """Find the banknote quadrilateral and return (aligned W x H image, found, quad)."""
    h, w = img.shape[:2]; s = min(1.0, 900 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0); total = g.size; best = None
    for mode in (0, 1):
        if mode == 0:
            m = cv2.morphologyEx(cv2.dilate(cv2.Canny(g, 40, 120), None, iterations=2), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        else:
            m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = sorted(cnts, key=cv2.contourArea, reverse=True)[:12]
        c = None
        for candidate in candidates:
            area = cv2.contourArea(candidate) / total
            if not 0.2 < area < 0.97: continue
            ap = cv2.approxPolyDP(candidate, 0.02 * cv2.arcLength(candidate, True), True)
            quad = ap.reshape(4, 2) if len(ap) == 4 else cv2.boxPoints(cv2.minAreaRect(candidate))
            if len(quad) == 4:
                c = candidate; break
        if c is None: continue
        area = cv2.contourArea(c) / total
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        quad = ap.reshape(4, 2) if len(ap) == 4 else cv2.boxPoints(cv2.minAreaRect(c))
        q = order_pts(quad); wd, ht = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
        if ht > wd: q = q[[3, 0, 1, 2]]; wd, ht = ht, wd
        if not 1.5 < wd / max(ht, 1) < 3.2: continue
        if best is None or area > best[0]: best = (area, q)
    if best is None:
        return cv2.resize(img, (W, H)), False, None
    q = best[1] / s
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype="float32"))
    return cv2.warpPerspective(img, M, (W, H)), True, q


def prep_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.GaussianBlur(g, (3, 3), 0))


def visibility(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    tex, edge, glare = float(g.std()), float((cv2.Canny(g, 50, 150) > 0).mean()), float((g >= 250).mean())
    if glare > CFG["max_glare"]: return False, "Strong glare covers much of this region."
    if tex < CFG["min_texture"] or edge < CFG["min_edge"]: return False, "Too little visible detail in this region (blank, dark or out of focus)."
    return True, ""


_refs = None
def load_refs():
    """Verified genuine reference notes, aligned to the canonical frame and cached."""
    global _refs
    if _refs is None:
        _refs = []
        for f in sorted(glob.glob(os.path.join(REF_DIR, "**", "*"), recursive=True)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                im = cv2.imread(f)
                if im is None: continue
                al, found, _ = detect_note(im)
                name = os.path.basename(f)
                side = "front" if "_front" in name.lower() else "back" if "_back" in name.lower() else "unknown"
                denomination = re.search(r"(?:rbi_)?(?:rs_)?(\d+)", name.lower())
                _refs.append(dict(name=name, img=al, aligned=found, side=side,
                                  denomination=denomination.group(1) if denomination else None,
                                  path=f))
    return _refs


_orb = cv2.ORB_create(2000, fastThreshold=10)
def orb_match(tg, rg):
    k1, d1 = _orb.detectAndCompute(tg, None); k2, d2 = _orb.detectAndCompute(rg, None)
    r = dict(kp_test=len(k1), kp_ref=len(k2), good=0, inliers=0, H=None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8: return r
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < CFG["ratio"] * p[1].distance]
    r["good"] = len(good)
    if len(good) >= 4:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if Hm is not None and mask is not None: r["inliers"], r["H"] = int(mask.sum()), Hm
    return r


def orient_orb_match(test_gray, reference_gray):
    """Try rotations and mirrored orientation; require enough strong geometry."""
    candidates = []
    for angle in (0, 90, 180, 270):
        if angle:
            rotated = np.rot90(reference_gray, angle // 90).copy()
        else:
            rotated = reference_gray
        candidate = orb_match(test_gray, rotated)
        candidates.append((candidate, rotated))
        mirrored = cv2.flip(rotated, 1)
        candidates.append((orb_match(test_gray, mirrored), mirrored))
    return max(candidates, key=lambda item: item[0]["inliers"] + item[0]["good"] / 10)


def ssim_stage(tg, rg, orb):
    a = cv2.resize(tg, (rg.shape[1], rg.shape[0]))
    if orb["H"] is not None and orb["inliers"] >= 8:
        a = cv2.warpPerspective(tg, orb["H"], (rg.shape[1], rg.shape[0]))
    s, smap = ssim_fn(a, rg, full=True, data_range=255)
    return float(s), np.clip(1 - smap, 0, 1)


_model = None
_anomaly = None
_distribution = None
_model_err = None


def load_ai_model(model_path=MODEL_PATH, names_path=None):
    """Load and validate the genuine-reference feature/anomaly model."""
    global _model, _anomaly, _distribution, _model_err
    _model = None
    _anomaly = None
    _distribution = None
    _model_err = None

    if not os.path.isfile(model_path):
        _model_err = f"AI model is unavailable: {model_path} was not found."
        return False, _model_err
    if not os.path.isfile(ANOMALY_PATH):
        _model_err = f"Reference anomaly model is unavailable: {ANOMALY_PATH} was not found."
        return False, _model_err
    if not os.path.isfile(DISTRIBUTION_PATH):
        _model_err = f"Genuine reference distribution is unavailable: {DISTRIBUTION_PATH} was not found."
        return False, _model_err

    try:
        from tensorflow import keras

        loaded_model = keras.models.load_model(model_path)
        input_shape = tuple(loaded_model.input_shape[1:3])
        if input_shape != (224, 224):
            raise ValueError(f"expected 224 x 224 model input, found {input_shape}.")
        loaded_anomaly = keras.models.load_model(ANOMALY_PATH)
        distribution = np.load(DISTRIBUTION_PATH)
        required = ("mean", "scale", "reference_embeddings",
                    "reconstruction_threshold", "distance_threshold")
        missing = [key for key in required if key not in distribution.files]
        if missing:
            raise ValueError(f"Reference distribution is missing: {', '.join(missing)}.")

        _model = loaded_model
        _anomaly = loaded_anomaly
        _distribution = distribution
        return True, None
    except Exception as exc:
        _model_err = f"AI model could not be loaded: {exc}"
        return False, _model_err


def model_predict(note):
    global _model, _anomaly, _distribution
    if _model is None:
        if _model_err is None:
            load_ai_model()
        return None, _model_err

    try:
        from tensorflow import keras
        rgb = cv2.cvtColor(cv2.resize(note, (224, 224)), cv2.COLOR_BGR2RGB)
        image = rgb.astype("float32")[None, ...]
        embedding = np.asarray(_model.predict(image, verbose=0)[0], dtype="float32")
        standardized = (embedding - _distribution["mean"]) / _distribution["scale"]
        reconstruction = np.asarray(
            _anomaly.predict(standardized[None, ...], verbose=0)[0], dtype="float32"
        )
        reconstruction_error = float(np.mean(np.square(reconstruction - standardized)))
        distance = float(np.sqrt(np.sum(np.square(standardized))))
        reconstruction_limit = float(_distribution["reconstruction_threshold"][0])
        distance_limit = float(_distribution["distance_threshold"][0])
        normalized = reconstruction_error / max(reconstruction_limit, 1e-6)
        distance_normalized = distance / max(distance_limit, 1e-6)
        if normalized <= 1 and distance_normalized <= 1:
            label = "Consistent with genuine references"
        elif normalized >= 20.0 and distance_normalized >= 2.5:
            label = "Unusual visual pattern"
        else:
            label = "Inconclusive"
        return dict(
            label=label,
            confidence=round(float(min(1, 1 / (1 + normalized))), 3),
            reconstruction_error=round(reconstruction_error, 6),
            distance=round(distance, 3),
            reconstruction_limit=round(reconstruction_limit, 6),
            distance_limit=round(distance_limit, 3),
            model_type="genuine_reference_anomaly_detection",
            disclaimer="This is visual reference comparison, not counterfeit detection. An unusual pattern does not prove counterfeiting.",
        ), None
    except Exception as exc:
        return None, f"AI prediction failed: {exc}"


def model_report():
    return json.load(open(EVAL_PATH)) if os.path.exists(EVAL_PATH) else None


def analyze(img, progress):
    refs = load_refs()
    progress(0, "Image received and decoded")
    q = quality(img); progress(1, "Checking brightness, blur and visibility")
    k = min(1, 1600 / max(img.shape[:2]))
    den = cv2.fastNlMeansDenoisingColored(cv2.resize(img, None, fx=k, fy=k), None, 3, 3, 7, 21)
    progress(2, "Detecting banknote boundary and aligning")
    note, found, quad = detect_note(den)
    compatible_refs = []
    if found and refs:
        target_gray = prep_gray(note)
        compatible_candidates = []
        for ref in refs:
            match, oriented_ref = orient_orb_match(target_gray, prep_gray(ref["img"]))
            if match["inliers"] >= 8 and ref["denomination"]:
                score = match["inliers"] + match["good"] / 10
                compatible_candidates.append((score, ref, oriented_ref))
        if compatible_candidates:
            compatible_candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, best_ref, _ = compatible_candidates[0]
            runner_score = compatible_candidates[1][0] if len(compatible_candidates) > 1 else 0
            if best_score - runner_score >= 3:
                compatible_refs = [
                    dict(ref) for ref in refs
                    if ref["denomination"] == best_ref["denomination"]
                ]
        refs = compatible_refs
    layout = [dict(id=r[0], name=r[1], box=r[2]) for r in REGIONS]
    progress(2, "Banknote aligned" if found else "Banknote boundary not found", extra=dict(preview=b64(note), layout=layout, found=found))
    progress(3, "Extracting corners and regions")
    items = []
    inspection_regions = REGIONS + FEATURE_REGIONS
    for rid, name, box, corner in inspection_regions:
        c = crop(note, box); ok, why = visibility(c)
        if not found: ok, why = False, "The banknote boundary could not be detected, so this region cannot be aligned reliably."
        items.append(dict(id=rid, name=name, box=box, corner=corner, crop=c, visible=ok, why=why, orb=None, ssim=None, diff=None, ref=None))
    progress(4, f"Matching against {len(refs)} verified reference image(s)")
    for it in items:
        if not (it["visible"] and refs): continue
        tg = prep_gray(it["crop"]); best = None
        for r in refs:
            rg = prep_gray(crop(r["img"], it["box"]))
            m, rg = orient_orb_match(tg, rg)
            if best is None or m["inliers"] + m["good"] / 10 > best[0]["inliers"] + best[0]["good"] / 10: best = (m, rg, r["name"])
        it["orb"], it["rg"], it["ref"], it["tg"] = best[0], best[1], best[2], tg
    progress(5, "Computing structural similarity and difference maps")
    for it in items:
        if it["orb"] is None: continue
        s, d = ssim_stage(it["tg"], it["rg"], it["orb"]); it["ssim"] = s
        it["diff"] = cv2.applyColorMap((d * 255).astype("uint8"), cv2.COLORMAP_INFERNO)
    progress(6, "Running MobileNetV2 classifier")
    mp, merr = model_predict(note)
    progress(7, "Building report")

    for it in items:
        o, s = it["orb"], it["ssim"]
        if not it["visible"]: it["status"], it["text"] = "yellow", it["why"]
        elif o is None: it["status"], it["text"] = "yellow", "No verified reference images are available in data/references, so this region cannot be compared."
        elif o["inliers"] >= CFG["inl_ok"] and s >= CFG["ssim_ok"]:
            it["status"], it["text"] = "green", "The extracted pattern is consistent with the verified reference region."
        elif o["inliers"] >= CFG["inl_bad"] and o["kp_test"] >= 12 and o["kp_ref"] >= 12 and s < CFG["ssim_bad"]:
            it["status"], it["text"] = "red", "The extracted pattern differs significantly from the corresponding reference region."
        else: it["status"], it["text"] = "yellow", "Evidence is mixed: the region partly matches the reference but not clearly enough to decide."
    OCR_REGION_IDS = {"top_left", "top_right", "bottom_left", "bottom_right", "serial_number"}
    for it in items:
        it["ocr"] = regional_text_assessment(it["crop"]) if it["id"] in OCR_REGION_IDS else None
    corners = [i for i in items if i["corner"]]
    red = [i for i in items if i["status"] == "red"]
    green_c = [i for i in corners if i["status"] == "green"]
    if q["level"] == "Poor":
        v = ("INCONCLUSIVE", "yellow", "Image quality or banknote detection is too poor for a reliable result. Retake the photo flat, upright, well lit and in focus.")
    elif mp and mp["label"] == "Consistent with genuine references":
        alignment_note = "The banknote boundary could not be detected, but the image passed the full-frame genuine-reference model check." if not found else "The banknote image matched the available genuine-reference distribution."
        v = ("LIKELY GENUINE", "green", f"{alignment_note} This is a visual screening result only, not proof of authenticity.")
    elif len(red) >= 2:
        v = ("INCONCLUSIVE", "yellow", f"{len(red)} inspected regions differ visually from the available references, but the genuine-reference model does not provide sufficient supporting evidence for a definite visual-screening result.")
    else:
        v = ("INCONCLUSIVE", "yellow", "Not enough consistent evidence across the reference distribution and inspected regions.")

    ann = note.copy(); n = 0
    for it in items:
        x0, y0, x1, y1 = it["box"]; p0, p1 = (int(x0 * W), int(y0 * H)), (int(x1 * W) - 2, int(y1 * H) - 2)
        cv2.rectangle(ann, p0, p1, COL[it["status"]], 4)
        if it["status"] == "red":
            n += 1; it["marker"] = n; cv2.circle(ann, (p0[0] + 26, p0[1] + 26), 22, COL["red"], -1)
            cv2.putText(ann, str(n), (p0[0] + 15, p0[1] + 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    assessed = [i for i in items if i["orb"] is not None]
    def avg(k):
        vals = [i["ssim"] if k == "ssim" else i["orb"][k] for i in assessed]
        return round(float(np.mean(vals)), 3) if vals else None
    regions = []
    for it in items:
        d = dict(id=it["id"], name=it["name"], corner=it["corner"], status=it["status"], text=it["text"], crop=b64(it["crop"]), marker=it.get("marker"),
                 visual_feature=REGION_FEATURES.get(it["id"], "Requires physical verification"))
        if it.get("ocr"):
            d["ocr"] = it["ocr"]
        if it["orb"]: d.update(orb=dict(good=it["orb"]["good"], inliers=it["orb"]["inliers"], kp=it["orb"]["kp_test"]), ssim=round(it["ssim"], 3), diff=b64(it["diff"]), ref=it["ref"])
        regions.append(d)
    return dict(
        verdict=dict(label=v[0], color=v[1], reason=v[2]), annotated=b64(ann), quality=q, found=found,
        alignment=dict(
            perspective_corrected=bool(found),
            boundary_detected=bool(found),
            status="Aligned" if found else "Original frame used because the boundary could not be detected reliably",
        ),
        overview=dict(model=mp, model_error=merr, model_report=model_report(), refs=len(refs),
                      orb=dict(mean_good=avg("good"), mean_inliers=avg("inliers")), ssim=avg("ssim"),
                      coverage=dict(assessed=len(assessed), total=len(items), corners=len([i for i in corners if i["orb"]]))),
        regions=        regions, suspicious=[r for r in regions if r["status"] == "red"], physical=PHYSICAL,
        inspection_criteria=VISUAL_INDICATORS)
