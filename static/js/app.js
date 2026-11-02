const $ = s => document.querySelector(s);
const SR = {
  stream: null,
  show(v) { document.querySelectorAll(".view").forEach(e => e.classList.add("d-none")); $("#view-" + v).classList.remove("d-none"); window.scrollTo(0, 0); },
  scrollSection(id) {
    if ($("#view-home").classList.contains("d-none")) SR.show("home");
    setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
  },
  err(msg) { const e = $("#home-err"); e.textContent = msg; e.classList.remove("d-none"); },
  pick(file) {
    $("#home-err").classList.add("d-none");
    if (!file) return;
    if (!/^image\/(jpeg|png)$/.test(file.type)) return SR.err("Only JPG, JPEG and PNG images are supported.");
    if (file.size > 12 * 1024 * 1024) return SR.err("Image is larger than 12 MB. Choose a smaller photo.");
    SR.scan(file);
  },
  async openCamera() {
    const m = bootstrap.Modal.getOrCreateInstance($("#camModal")); const ce = $("#cam-err"); ce.classList.add("d-none");
    $("#cam-status").textContent = "Starting live camera…"; $("#cam-status").classList.remove("d-none"); m.show();
    if (!navigator.mediaDevices?.getUserMedia) {
      ce.textContent = "Live camera requires a supported browser and a secure connection (localhost or HTTPS).";
      ce.classList.remove("d-none"); $("#cam-status").classList.add("d-none"); return;
    }
    try {
      SR.stopCamera();
      SR.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
      $("#video").srcObject = SR.stream;
      await $("#video").play();
      $("#cam-status").textContent = "● Live camera active";
    }
    catch (e) {
      ce.textContent = "Camera unavailable or permission denied. Allow camera access in the browser, or upload a photo instead.";
      ce.classList.remove("d-none"); $("#cam-status").classList.add("d-none");
    }
  },
  stopCamera() { if (SR.stream) SR.stream.getTracks().forEach(t => t.stop()); SR.stream = null; },
  async scan(file) {
    SR.show("scan");
    const box = $(".scanbox"); box.classList.remove("done"); $("#boxes").innerHTML = ""; $("#scan-err").classList.add("d-none"); $("#retry").classList.add("d-none");
    $("#scan-img").src = URL.createObjectURL(file); SR.paint(-1, "Uploading");
    const fd = new FormData(); fd.append("image", file, file.name || "capture.jpg");
    let job;
    try { const r = await fetch("/api/analyze", { method: "POST", body: fd }); job = await r.json(); if (!r.ok) throw new Error(job.error); }
    catch (e) { return SR.fail(e.message || "Upload failed."); }
    let st = null, shown = 0, boxed = false, finished = false;
    const poll = setInterval(async () => {
      try { const r = await fetch("/api/status/" + job.job_id); st = await r.json(); if (!r.ok) throw new Error(st.error); }
      catch (e) { clearInterval(poll); clearInterval(tick); SR.fail(e.message); }
    }, 300);
    // The UI follows the backend's real stage; each stage is shown for at least 500 ms so it is readable.
    const tick = setInterval(() => {
      if (!st) return;
      if (st.error) { clearInterval(poll); clearInterval(tick); return SR.fail(st.error); }
      if (st.preview && shown >= 2 && !boxed) { boxed = true; $("#scan-img").src = st.preview; SR.boxes(st.layout); }
      if (shown < st.stage) shown++;
      SR.paint(shown, st.detail);
      if (st.layout && shown >= 3) document.querySelectorAll(".rbox").forEach((b, i) => b.style.opacity = shown >= 3 + 0 ? 1 : 0);
      if (st.done && shown >= st.stage && !finished) { finished = true; clearInterval(poll); clearInterval(tick); SR.paint(8, "Complete"); box.classList.add("done"); setTimeout(() => SR.render(st.result), 700); }
    }, 500);
  },
  paint(i, detail) {
    document.querySelectorAll("#stages li").forEach((li, k) => { li.className = k < i ? "done" : k === i ? "active" : ""; });
    const p = Math.round(Math.max(0, i) / 8 * 100); $("#bar").style.width = p + "%"; $("#pct").textContent = p + "%"; $("#detail").textContent = detail || "";
  },
  boxes(layout) {
    $("#boxes").innerHTML = layout.map(r => `<div class="rbox" title="${r.name}" style="left:${r.box[0]*100}%;top:${r.box[1]*100}%;width:${(r.box[2]-r.box[0])*100}%;height:${(r.box[3]-r.box[1])*100}%"></div>`).join("");
  },
  fail(msg) { $("#scan-err").textContent = msg; $("#scan-err").classList.remove("d-none"); $("#retry").classList.remove("d-none"); document.querySelector(".scanbox").classList.add("done"); },
  render(r) {
    const o = r.overview, v = r.verdict, f = (x, d = 2) => x == null ? "n/a" : (+x).toFixed(d);
    const m = o.model ? `<b>${(o.model.confidence * 100).toFixed(1)}%</b><span>${o.model.label}${o.model_report ? ` · held-out test accuracy ${(o.model_report.accuracy * 100).toFixed(1)}%` : " · not evaluated yet"}</span>` : `<b>n/a</b><span>${o.model_error || ""}</span>`;
    const st = { green: "Reference consistent", red: "Suspicious mismatch", yellow: "Inconclusive" };
    const card = x => `<div class="col-md-6 col-xl-4">
      <article class="panel detailed-region-card ${x.status}">
        <div class="detailed-region-header">
          <div><span>REGION ${x.marker || "•"}</span><b>${x.name}</b><small>${x.visual_feature || ""}</small></div>
          <span class="detail-status ${x.status}">${x.status === "green" ? "Consistent" : x.status === "red" ? "Suspicious" : "Inconclusive"}</span>
        </div>
        <div class="detail-images">
          <figure><img src="${x.crop}" alt="${x.name}"><figcaption>Extracted region</figcaption></figure>
          ${x.diff ? `<figure><img src="${x.diff}" alt="${x.name} difference map"><figcaption>Difference map</figcaption></figure>` : ""}
        </div>
        <p class="detail-finding">${x.text}</p>
        ${x.ocr ? `<div class="detail-unavailable">OCR visual text evidence: <b>${x.ocr.status}</b>${x.ocr.text_confidence == null ? "" : ` (confidence ${f(x.ocr.text_confidence * 100, 1)}%)`}. OCR does not verify authenticity or identify denomination.</div>` : ""}
        ${x.orb ? `<div class="detail-metrics">
          <div><span>ORB matches</span><b>${x.orb.good}</b></div>
          <div><span>Inliers</span><b>${x.orb.inliers}</b></div>
          <div><span>SSIM</span><b>${f(x.ssim)}</b></div>
        </div><div class="detail-reference">Best reference <b>${x.ref}</b></div>`
          : `<div class="detail-unavailable">No verified reference data available for this region.</div>`}
      </article>
    </div>`;
    const sus = r.suspicious.length ? r.suspicious.map(x => `<div class="panel rcard red mb-3"><div class="row g-3 align-items-center"><div class="col-md-3"><img src="${x.crop}" class="w-100 rounded"></div><div class="col-md-3"><img src="${x.diff}" class="w-100 rounded"></div>
      <div class="col-md-6"><b>Suspicious Region: ${x.name}</b><p class="mb-1">Finding: ${x.text}</p><p class="mb-1">Status: <span class="pill red">SUSPICIOUS</span></p><div class="kv">ORB inliers <b>${x.orb.inliers}</b> of <b>${x.orb.good}</b> good matches · SSIM <b>${f(x.ssim)}</b> (bright areas in the map mark the largest differences)</div></div></div></div>`).join("")
      : `<div class="panel">No region was flagged as a suspicious reference mismatch. ${v.color === "green" ? "Visible patterns are consistent with the available references; this is not physical proof of authenticity." : "Some regions could not be assessed, so the result is inconclusive."}</div>`;
    const attention = r.regions.filter(x => x.status !== "green");
    const attentionHtml = attention.length
      ? `<div class="attention-region-grid">${attention.map((x, index) => `
        <article class="attention-region ${x.status}">
          <div class="attention-region-image">
            <span class="attention-index">${String(index + 1).padStart(2, "0")}</span>
            <img src="${x.crop}" alt="${x.name}">
          </div>
          <div class="attention-region-content">
            <div class="attention-region-heading">
              <b>${x.name}</b>
              <small>${x.visual_feature || ""}</small>
              <span class="${x.status}">${st[x.status]}</span>
            </div>
            <p>${x.text}</p>
            <div class="attention-region-metrics">
              <div><span>ORB</span><b>${x.orb ? x.orb.inliers : "n/a"}</b></div>
              <div><span>SSIM</span><b>${x.ssim == null ? "n/a" : f(x.ssim)}</b></div>
            </div>
          </div>
        </article>`).join("")}</div>`
      : `<p class="mb-0">No suspicious or inconclusive region was reported.</p>`;
    const genuine = r.regions.filter(x => x.status === "green").length;
    const suspicious = r.regions.filter(x => x.status === "red").length;
    const inconclusive = r.regions.filter(x => x.status === "yellow").length;
    const confidence = o.model ? Math.round(o.model.confidence * 100) : null;
    const authenticity = v.color === "green"
      ? { label: "LIKELY GENUINE", detail: "Visual pattern matches the available genuine references" }
      : v.color === "red"
        ? { label: "UNUSUAL VISUAL PATTERN", detail: "This does not prove counterfeiting" }
        : { label: "INCONCLUSIVE", detail: "The available evidence is insufficient for a reliable visual-screening result" };
    const modelStatus = o.model
      ? `<b>${confidence}%</b><span>${o.model.label} prediction</span>`
      : `<b>Unavailable</b><span class="technical-error">${o.model_error || "AI model was not loaded."}</span>`;
    $("#view-result").innerHTML = `
      <header class="overview-heading text-center"><h1><span>SmartRupee – AI</span> Genuine Reference Analysis</h1><p>Upload an Indian banknote image for visual screening against available genuine references. Unusual patterns do not prove counterfeiting.</p></header>
      <section class="panel final-result-banner ${v.color}" role="status" aria-live="polite">
        <span>Final visual screening result</span>
        <strong>${authenticity.label}</strong>
        <p>${authenticity.detail}. ${v.reason}</p>
      </section>
      <div class="results-dashboard">
        <section class="result-center">
          <div class="panel result-note"><h2>▧ &nbsp;Uploaded Currency</h2><img src="${r.annotated}" alt="Analyzed banknote"></div>
          <div class="panel region-overview"><h2>Region-wise Reference Analysis</h2><div class="region-strip">${r.regions.map(x => `<div class="region-card"><img src="${x.crop}" alt="${x.name}"><b>${x.name}</b><strong>${x.status === "green" ? "Reference Consistent" : x.status === "red" ? "Visual Mismatch" : "Inconclusive"}</strong></div>`).join("")}</div></div>
        </section>
        <aside class="panel analysis-panel"><h2>🛡 &nbsp;Analysis Result</h2>
          <div class="authenticity-result ${v.color}">
            <span>${authenticity.label}</span>
            <small>${authenticity.detail}</small>
          </div>
          <div class="verdict compact ${v.color}"><strong>${v.label}</strong><small>${v.reason}</small></div>
          <div class="score-overview"><h3>AI Confidence</h3><div class="score-ring" style="--score:${confidence == null ? 0 : confidence * 3.6}deg"><div>${modelStatus}</div></div></div>
          <div class="mini-scores">
            <div><span>ORB inliers</span><b>${o.orb.mean_inliers == null ? "Unavailable" : f(o.orb.mean_inliers, 1)}</b></div>
            <div><span>SSIM similarity</span><b>${o.ssim == null ? "Unavailable" : f(o.ssim)}</b></div>
            <div><span>Image quality</span><b>${r.quality.score}/100</b></div>
            <div><span>Region coverage</span><b>${o.coverage.assessed}/${o.coverage.total}</b></div>
          </div>
          <div class="detection-summary"><h3>Detection Summary</h3><div><span class="summary-icon green">✓</span><span>Genuine Regions</span><b>${genuine}</b></div><div><span class="summary-icon red">!</span><span>Suspicious Regions</span><b>${suspicious}</b></div><div><span class="summary-icon yellow">⚠</span><span>Inconclusive Regions</span><b>${inconclusive}</b></div></div>
          <div class="key-findings"><h3>Key Findings</h3><p>• Image quality: ${r.quality.level} (${r.quality.score}/100), blur variance ${f(r.quality.blur, 1)}.</p>${r.quality.issues.length ? r.quality.issues.map(x => `<p>• ${x}</p>`).join("") : "<p>• Lighting and sharpness are adequate for visual analysis.</p>"}<p>• ${o.model ? "AI prediction is one screening signal, not a physical authenticity test." : "No AI prediction was generated because the model was unavailable."}</p><p>• Similarity scores are not authenticity percentages.</p></div>
          <button class="btn btn-glow w-100 mt-3" onclick="SR.show('home')">Scan another note</button>
        </aside>
      </div>
      <section class="panel mt-4 attention-panel"><h3>Highlighted Suspicious or Inconclusive Regions</h3>${attentionHtml}</section>
      <section class="panel detail-results mt-4"><h3>Detailed Region Analysis</h3><div class="row g-3">${r.regions.map(card).join("")}</div></section>
      <section class="panel mt-4"><h3>Visual Counterfeit Indicators</h3><p>These indicators are inspection criteria, not proof of counterfeiting. Missing evidence is inconclusive and may vary by denomination, series, wear and lighting.</p><div class="indicator-grid">${r.inspection_criteria.map(x => `<div class="panel indicator-card"><b>${x.name}</b><p class="mb-0">${x.assessment}</p></div>`).join("")}</div></section>
      <section class="panel mt-4"><h3>Requires Physical Verification</h3><p>Normal RGB photographs cannot verify tactile raised-print, UV fluorescence, infrared response, transmitted-light alignment, colour-shift ink or substrate quality.</p><ul class="mb-0">${r.physical.map(p => `<li>${p}</li>`).join("")}</ul><small>Counterfeit classification requires independently verified genuine and counterfeit examples, not this indicators list alone.</small></section>`;
    SR.show("result");
  }
};
$("#file").addEventListener("change", e => { SR.pick(e.target.files[0]); e.target.value = ""; });
const drop = $("#drop");
["dragover", "dragenter"].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => SR.pick(e.dataTransfer.files[0]));
drop.addEventListener("click", () => $("#file").click());
$("#snap").addEventListener("click", () => {
  const v = $("#video"); if (!v.videoWidth) return;
  const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob(b => { bootstrap.Modal.getInstance($("#camModal")).hide(); SR.scan(new File([b], "capture.jpg", { type: "image/jpeg" })); }, "image/jpeg", 0.95);
});
$("#camModal").addEventListener("hidden.bs.modal", SR.stopCamera);
