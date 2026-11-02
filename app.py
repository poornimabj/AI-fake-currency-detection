import io, time, uuid, threading, traceback
import cv2, numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, UnidentifiedImageError
from smartrupee.engine import analyze, load_ai_model, STAGES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
JOBS = {}

MODEL_LOADED, MODEL_ERROR = load_ai_model()
if MODEL_LOADED:
    print("SmartRupee AI model loaded successfully.")
else:
    print(MODEL_ERROR)


def run(jid, img):
    job = JOBS[jid]
    def progress(i, detail, extra=None):
        job.update(stage=i, detail=detail)
        if extra: job.update(extra)
    try:
        job["result"] = analyze(img, progress); job["stage"] = len(STAGES) - 1; job["done"] = True
    except Exception as e:
        traceback.print_exc(); job.update(error=f"Analysis failed: {e}", done=True)


@app.route("/")
def index():
    return render_template("index.html", stages=STAGES)


@app.post("/api/analyze")
def start():
    f = request.files.get("image")
    if not f: return jsonify(error="No image received."), 400
    try:
        im = Image.open(io.BytesIO(f.read()))
        if im.format not in ("JPEG", "PNG"): return jsonify(error="Only JPG, JPEG and PNG images are supported."), 400
        im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="The file is not a readable image."), 400
    for k in [k for k, j in JOBS.items() if time.time() - j["t"] > 1800]: JOBS.pop(k)
    jid = uuid.uuid4().hex
    JOBS[jid] = dict(t=time.time(), stage=0, detail="Queued", done=False, error=None, result=None, preview=None, layout=None, found=None)
    threading.Thread(target=run, args=(jid, cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)), daemon=True).start()
    return jsonify(job_id=jid)


@app.get("/api/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j: return jsonify(error="Unknown or expired scan."), 404
    return jsonify({k: j[k] for k in ("stage", "detail", "done", "error", "preview", "layout", "found")} | ({"result": j["result"]} if j["done"] else {}))


@app.errorhandler(413)
def too_big(_): return jsonify(error="Image is larger than 12 MB."), 413


if __name__ == "__main__":
    app.run(debug=False, port=5000)
