// Robo Judge v5: all metrics in dashboard, skeleton only on video, barbell tracking & velocity
const video = document.getElementById('video');
const stage = document.getElementById('stage');
const ctx = stage.getContext('2d');
const tol = document.getElementById('tol'); const tolVal = document.getElementById('tolVal');
const smooth = document.getElementById('smooth'); const smVal = document.getElementById('smVal');
const cameraSel = document.getElementById('camera'); const flipBtn = document.getElementById('flip');
const accSel = document.getElementById('accuracy'); const fileInput = document.getElementById('file');
const recBtn = document.getElementById('recBtn'); const stopBtn = document.getElementById('stopBtn'); const dlLink = document.getElementById('dlLink');
const barbellBtn = document.getElementById('barbellBtn');

// Dashboard fields
const depthStatus = document.getElementById('depthStatus');
const marginEl = document.getElementById('margin');
const kneeAnglesEl = document.getElementById('kneeAngles');
const hipAnglesEl = document.getElementById('hipAngles');
const velInstantEl = document.getElementById('velInstant');
const velPeakEl = document.getElementById('velPeak');
const statusEl = document.getElementById('status');

let currentDeviceId = null; let useFront = false; let processingVideoFile = false;
let recorder = null; let recordedChunks = [];
let frameCount = 0; let noPose = 0;
const hipHist = [], kneeHist = [];
let cmPerPx = 0.2;

// Barbell tracking state
let trackingEnabled = false;
let template = null; // ImageData of barbell patch
let templatePos = null; // {x,y} center (normalized 0..1)
let lastBarbellY = null; // in pixels
let lastTime = null;
let peakVel = 0;

// Helpers
function setStageToVideo() {
  const vw = video.videoWidth || 720, vh = video.videoHeight || 1280;
  stage.width = vw; stage.height = vh;
  stage.style.width = "100%"; stage.style.height = "100%";
}
function pushHist(hist, v, max){ hist.push(v); if(hist.length>max) hist.shift(); }
function avg(arr){ return arr.reduce((s,v)=>s+v,0)/Math.max(arr.length,1); }
function dist(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy); }
function dot(ax,ay,bx,by){ return ax*bx+ay*by; }
function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

function estimateCmPerPx(landmarks) {
  const lk = landmarks[25], la = landmarks[27], rk = landmarks[26], ra = landmarks[28];
  if(!lk||!la||!rk||!ra) return cmPerPx || 0.2;
  const l1 = dist(lk,la), l2 = dist(rk,ra);
  const lower = (l1+l2)/2;
  const tibiaCm = 38.5;
  const lowerPx = lower * stage.height;
  cmPerPx = tibiaCm / Math.max(lowerPx, 1);
  return cmPerPx;
}

function angleDeg(a,b,c){ // angle at b (in degrees), using normalized coords
  if(!a||!b||!c) return null;
  const ABx = a.x - b.x, ABy = a.y - b.y;
  const CBx = c.x - b.x, CBy = c.y - b.y;
  const dotp = dot(ABx,ABy,CBx,CBy);
  const mag1 = Math.hypot(ABx,ABy), mag2 = Math.hypot(CBx,CBy);
  if (mag1 === 0 || mag2 === 0) return null;
  const cos = clamp(dotp/(mag1*mag2), -1, 1);
  return Math.round((Math.acos(cos) * 180 / Math.PI));
}

function judgeDepth(landmarks) {
  const lh = landmarks[23], rh = landmarks[24], lk = landmarks[25], rk = landmarks[26];
  if(!lh||!rh||!lk||!rk) return null;
  const hipY = Math.min(lh.y, rh.y);  // lower hip
  const kneeY = Math.max(lk.y, rk.y); // top of knees
  pushHist(hipHist, hipY, +smooth.value);
  pushHist(kneeHist, kneeY, +smooth.value);
  const hipYsm = avg(hipHist), kneeYsm = avg(kneeHist);
  const tolY = +tol.value;
  const isGood = (hipYsm <= kneeYsm + tolY);
  const marginNorm = (kneeYsm - hipYsm);
  const marginPx = marginNorm * stage.height;
  const cmpp = estimateCmPerPx(landmarks);
  const marginCm = marginPx * cmpp;
  const depthPct = Math.max(0, Math.min(1.2, marginCm / 4.0));
  return {isGood, marginCm, depthPct};
}

// Skeleton drawing (no text on video; color only)
function drawSkeleton(landmarks, isGood) {
  if(!landmarks) return;
  const px = (p)=>({x: p.x*stage.width, y: p.y*stage.height});
  const pairs = [
    [11,13],[13,15], [12,14],[14,16],
    [23,25],[25,27], [24,26],[26,28],
    [11,12],[23,24],
    [11,23],[12,24]
  ];
  ctx.lineWidth = 3;
  ctx.strokeStyle = isGood ? "rgba(90,245,118,0.9)" : "rgba(255,107,107,0.95)";
  pairs.forEach(([a,b])=>{
    const pa = landmarks[a], pb = landmarks[b]; if(!pa||!pb) return;
    const A = px(pa), B = px(pb);
    ctx.beginPath(); ctx.moveTo(A.x,A.y); ctx.lineTo(B.x,B.y); ctx.stroke();
  });
  ctx.fillStyle = "#ffffff";
  landmarks.forEach(p=>{
    if(!p) return;
    const P = px(p);
    ctx.beginPath(); ctx.arc(P.x, P.y, 4, 0, Math.PI*2); ctx.fill();
  });
}

// Barbell tracking: simple template matching within a window
function getPatch(x,y,sz){
  const ix = Math.round(x - sz/2), iy = Math.round(y - sz/2);
  const w = Math.round(sz), h = Math.round(sz);
  if (ix<0||iy<0||ix+w>stage.width||iy+h>stage.height) return null;
  return ctx.getImageData(ix, iy, w, h);
}
function matchPatchAround(x,y,sz,rad){
  // Search within a square window of radius rad around (x,y) for best correlation
  let bestScore = -1, best = {x, y};
  const target = template;
  if (!target) return best;
  for(let dy=-rad; dy<=rad; dy+=2){
    for(let dx=-rad; dx<=rad; dx+=2){
      const cx = Math.round(x+dx), cy = Math.round(y+dy);
      const cand = getPatch(cx, cy, sz);
      if (!cand) continue;
      // NCC-like score (very simplified): sum of absolute differences inverted
      let s = 0, n = 0;
      const ta = target.data, ca = cand.data;
      for(let i=0; i<ta.length; i+=16){ // subsample for speed
        const dr = Math.abs(ta[i]-ca[i]);
        const dg = Math.abs(ta[i+1]-ca[i+1]);
        const db = Math.abs(ta[i+2]-ca[i+2]);
        s += (255*3 - (dr+dg+db));
        n++;
      }
      const score = s / Math.max(n,1);
      if (score > bestScore){ bestScore = score; best = {x:cx, y:cy}; }
    }
  }
  return best;
}

function updateBarbellVelocity(yPx){
  const now = performance.now();
  if (lastTime == null){ lastTime = now; lastBarbellY = yPx; return {v:0,peak:peakVel}; }
  const dt = (now - lastTime) / 1000.0; // seconds
  const dy = (yPx - lastBarbellY);      // pixels (down is +)
  const v_cms = -(dy * cmPerPx) / dt;   // negative dy (upwards) => positive velocity upward
  const v_ms = v_cms / 100.0;
  lastTime = now; lastBarbellY = yPx;
  // peak per rep (simple heuristic: reset when velocity sign changes from positive to negative)
  if (!Number.isFinite(v_ms)) return {v:0,peak:peakVel};
  if (v_ms > peakVel) peakVel = v_ms;
  if (Math.sign(v_ms) < 0 && peakVel>0.001) { // end of concentric
    peakVel = 0; // reset for next rep
  }
  return {v:v_ms, peak:peakVel};
}

let awaitingTemplateClick = false;
barbellBtn.addEventListener('click', ()=>{
  awaitingTemplateClick = true;
  statusEl.textContent = "Tap the barbell knurl to initialize tracking…";
  trackingEnabled = false;
});

stage.addEventListener('click', (e)=>{
  if (!awaitingTemplateClick) return;
  const rect = stage.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (stage.width / rect.width);
  const y = (e.clientY - rect.top) * (stage.height / rect.height);
  // capture template around click
  const sz = Math.max(24, Math.round(stage.width*0.06));
  const patch = getPatch(x,y,sz);
  if (patch){
    template = patch;
    templatePos = {x, y, sz};
    trackingEnabled = true;
    awaitingTemplateClick = false;
    statusEl.textContent = "Barbell tracking initialized.";
  } else {
    statusEl.textContent = "Template out of bounds. Tap again.";
  }
});

function drawBarbellMarker(pos){
  if (!pos) return;
  ctx.save();
  ctx.strokeStyle = "rgba(255,215,0,0.95)";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(pos.x, pos.y, Math.max(12, pos.sz/2), 0, Math.PI*2); ctx.stroke();
  ctx.restore();
}

function drawFrame(landmarks, verdict){
  // draw video
  ctx.clearRect(0,0,stage.width,stage.height);
  ctx.drawImage(video, 0, 0, stage.width, stage.height);

  // skeleton
  drawSkeleton(landmarks, verdict?.isGood);

  // barbell tracking
  if (trackingEnabled && template && templatePos){
    // track around last pos
    const searchRad = Math.round(Math.max(20, stage.height*0.05));
    const found = matchPatchAround(templatePos.x, templatePos.y, templatePos.sz, searchRad);
    if (found){
      templatePos.x = found.x; templatePos.y = found.y;
      const vel = updateBarbellVelocity(found.y);
      velInstantEl.textContent = `${vel.v.toFixed(2)} m/s`;
      velPeakEl.textContent = `${vel.peak.toFixed(2)} m/s`;
    }
    drawBarbellMarker(templatePos);
  }
}

// Camera / video plumbing + recorder
async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  cameraSel.innerHTML = '';
  const vids = devices.filter(d=>d.kind==='videoinput');
  vids.forEach((d,i)=>{
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera ${i+1}`;
    cameraSel.appendChild(opt);
  });
  if (vids.length && !currentDeviceId) currentDeviceId = vids[0].deviceId;
  if (currentDeviceId) cameraSel.value = currentDeviceId;
}
cameraSel.addEventListener('change', async ()=>{
  currentDeviceId = cameraSel.value;
  if (!processingVideoFile) await startCamera();
});
flipBtn.addEventListener('click', async ()=>{
  useFront = !useFront;
  if (!processingVideoFile) await startCamera();
});
fileInput.addEventListener('change', async (e)=>{
  const file = e.target.files[0]; if (!file) return;
  processingVideoFile = true;
  // stop camera
  if (video.srcObject) { video.srcObject.getTracks().forEach(t=>t.stop()); video.srcObject = null; }
  video.src = URL.createObjectURL(file);
  video.muted = true;
  await video.play();
  setStageToVideo();
  startRecorder();
  statusEl.textContent = "Analyzing video file… tap 🔴 REC to capture overlay if desired.";
});

function startRecorder() {
  if (recorder) { try { recorder.stop(); } catch(e){} recorder = null; }
  const stream = stage.captureStream(30);
  recordedChunks = [];
  try {
    recorder = new MediaRecorder(stream, {mimeType: 'video/webm;codecs=vp9'});
  } catch(e) {
    recorder = new MediaRecorder(stream);
  }
  recorder.ondataavailable = (e)=>{ if (e.data.size) recordedChunks.push(e.data); };
  recorder.onstop = ()=>{
    const blob = new Blob(recordedChunks, {type: recordedChunks[0]?.type || 'video/webm'});
    const url = URL.createObjectURL(blob);
    dlLink.href = url; dlLink.style.display = 'inline'; dlLink.textContent = 'Download recording';
  };
}
recBtn.addEventListener('click', ()=>{
  if (!recorder) startRecorder();
  recorder.start();
  recBtn.classList.add('recording'); recBtn.disabled = true; stopBtn.disabled = false; dlLink.style.display = 'none';
});
stopBtn.addEventListener('click', ()=>{
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  recBtn.classList.remove('recording'); recBtn.disabled = false; stopBtn.disabled = true;
});

tol.addEventListener('input', ()=> tolVal.textContent = (+tol.value).toFixed(3));
smooth.addEventListener('input', ()=> smVal.textContent = smooth.value);

async function startCamera() {
  processingVideoFile = false;
  if (!navigator.mediaDevices?.getUserMedia) { alert('Camera not supported'); return; }
  const constraints = {
    audio:false,
    video: currentDeviceId ? {deviceId: {exact: currentDeviceId}} : {facingMode: useFront ? 'user' : 'environment'}
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream; video.muted = true;
  await video.play();
  setStageToVideo();
  startRecorder();
  statusEl.textContent = "Live mode ready.";
}

function computeAngles(lm) {
  const Lknee = angleDeg(lm[23], lm[25], lm[27]);
  const Rknee = angleDeg(lm[24], lm[26], lm[28]);
  const Lhip = angleDeg(lm[11], lm[23], lm[25]);
  const Rhip = angleDeg(lm[12], lm[24], lm[26]);
  kneeAnglesEl.textContent = `${Lknee??"—"}° / ${Rknee??"—"}°`;
  hipAnglesEl.textContent = `${Lhip??"—"}° / ${Rhip??"—"}°`;
}

function loopPose(pose){
  if (video.readyState >= 2) {
    pose.send({image: video}).then(()=>{
      requestAnimationFrame(()=>loopPose(pose));
    });
  } else {
    requestAnimationFrame(()=>loopPose(pose));
  }
}

async function main() {
  await listCameras();
  await startCamera();

  const pose = new Pose.Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`
  });
  pose.setOptions({
    modelComplexity: +accSel.value,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  pose.onResults((res)=>{
    frameCount++;
    setStageToVideo();
    const lm = res.poseLandmarks;
    if (!lm) {
      noPose++;
      drawFrame(null, null);
      depthStatus.textContent = "No pose detected";
      statusEl.textContent = `No pose… step back / include hips+knees • NoPose frames: ${noPose}`;
      marginEl.textContent = "— cm";
      return;
    }
    noPose = 0;
    const verdict = judgeDepth(lm);
    drawFrame(lm, verdict);
    computeAngles(lm);

    if (verdict) {
      depthStatus.textContent = verdict.isGood ? "GOOD DEPTH" : "NOT DEEP ENOUGH";
      depthStatus.className = verdict.isGood ? "good" : "fail";
      marginEl.textContent = `${verdict.marginCm.toFixed(1)} cm`;
      statusEl.textContent = "Tracking…";
    }
  });

  accSel.addEventListener('change', ()=>{
    pose.setOptions({modelComplexity: +accSel.value});
  });

  loopPose(pose);
}

main().catch(e=>{
  statusEl.textContent = "Error: " + e.message;
});
