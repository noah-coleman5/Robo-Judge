// Robo Judge v2 — improved tracking & debug
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const call = document.getElementById('call');
const detail = document.getElementById('detail');
const score = document.getElementById('score');
const debug = document.getElementById('debug');
const tol = document.getElementById('tol');
const tolVal = document.getElementById('tolVal');
const smooth = document.getElementById('smooth');
const smVal = document.getElementById('smVal');
const cameraSel = document.getElementById('camera');
const flipBtn = document.getElementById('flip');
const accSel = document.getElementById('accuracy');
const fileInput = document.getElementById('file');

let currentDeviceId = null;
let useFront = false;
let frameCount = 0;
let noPoseFrames = 0;
let processingVideoFile = false;

function setCanvasToVideo() {
  // Use actual video pixels to keep landmark mapping accurate
  const vw = video.videoWidth || 720;
  const vh = video.videoHeight || 1280;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(vw * dpr);
  canvas.height = Math.round(vh * dpr);
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener('resize', setCanvasToVideo);

tol.addEventListener('input', ()=> tolVal.textContent = (+tol.value).toFixed(3));
smooth.addEventListener('input', ()=> smVal.textContent = smooth.value);

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

accSel.addEventListener('change', ()=>{
  // handled in pose setup
});

fileInput.addEventListener('change', async (e)=>{
  const file = e.target.files[0];
  if (!file) return;
  processingVideoFile = true;
  // Stop any camera tracks
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t=>t.stop());
    video.srcObject = null;
  }
  video.src = URL.createObjectURL(file);
  await video.play();
  setCanvasToVideo();
});

async function startCamera() {
  processingVideoFile = false;
  if (!navigator.mediaDevices?.getUserMedia) {
    alert('Camera not supported in this browser.');
    return;
  }
  const constraints = {
    audio:false,
    video: currentDeviceId ? {deviceId: {exact: currentDeviceId}} : {facingMode: useFront ? 'user' : 'environment'}
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  setCanvasToVideo();
}

function drawSkeleton(landmarks) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!landmarks) return;
  const pairs = [
    [11,13],[13,15], [12,14],[14,16],
    [23,25],[25,27], [24,26],[26,28],
    [11,12],[23,24],
    [11,23],[12,24]
  ];
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(120,180,255,0.9)';
  pairs.forEach(([a,b])=>{
    const pa = landmarks[a], pb = landmarks[b];
    if(!pa||!pb) return;
    ctx.beginPath();
    ctx.moveTo(pa.x*canvas.width, pa.y*canvas.height);
    ctx.lineTo(pb.x*canvas.width, pb.y*canvas.height);
    ctx.stroke();
  });
  // draw keypoints
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  landmarks.forEach(p=>{
    if(!p) return;
    ctx.beginPath();
    ctx.arc(p.x*canvas.width, p.y*canvas.height, 3, 0, Math.PI*2);
    ctx.fill();
  });
}

function pushHist(hist, v, max){ hist.push(v); if(hist.length>max) hist.shift(); }
function avg(arr){ return arr.reduce((s,v)=>s+v,0)/Math.max(1,arr.length); }

const hipHist = [], kneeHist = [];

function estimateCmPerPx(landmarks) {
  const lk = landmarks[25], la = landmarks[27];
  const rk = landmarks[26], ra = landmarks[28];
  if(!lk||!la||!rk||!ra) return 0.2;
  const dx1 = (lk.x - la.x) * canvas.width;
  const dy1 = (lk.y - la.y) * canvas.height;
  const dx2 = (rk.x - ra.x) * canvas.width;
  const dy2 = (rk.y - ra.y) * canvas.height;
  const l1 = Math.hypot(dx1,dy1);
  const l2 = Math.hypot(dx2,dy2);
  const lower = (l1+l2)/2;
  const tibiaCm = 38.5;
  return tibiaCm / Math.max(lower, 1);
}

function judge(landmarks) {
  const lh = landmarks[23], rh = landmarks[24];
  const lk = landmarks[25], rk = landmarks[26];
  if(!lh||!rh||!lk||!rk) return null;

  // MediaPipe: origin top-left, y grows down
  const hipY = Math.min(lh.y, rh.y);
  const kneeY = Math.max(lk.y, rk.y);
  pushHist(hipHist, hipY, +smooth.value);
  pushHist(kneeHist, kneeY, +smooth.value);
  const hipYsm = avg(hipHist), kneeYsm = avg(kneeHist);

  const tolY = +tol.value;
  const isGood = (hipYsm <= kneeYsm + tolY);

  const marginNorm = (kneeYsm - hipYsm);
  const cmPerPx = estimateCmPerPx(landmarks);
  const marginPx = marginNorm * canvas.height;
  const marginCm = marginPx * cmPerPx;
  const depthPct = Math.max(0, Math.min(1.2, marginCm / 4.0));

  return {isGood, marginCm, depthPct};
}

async function setupPose() {
  const pose = new Pose.Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`
  });
  pose.setOptions({
    modelComplexity: +accSel.value, // 1 fast, 2 high
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  pose.onResults((res)=>{
    frameCount++;
    setCanvasToVideo();
    const lm = res.poseLandmarks;
    if (!lm) {
      noPoseFrames++;
      drawSkeleton(null);
      call.textContent = "No pose detected… step back / include hips & knees";
      call.className = "";
      detail.textContent = "Hip–Knee: -- cm";
      score.textContent = "Depth: -- %";
      debug.textContent = `Frames: ${frameCount} • NoPose: ${noPoseFrames}`;
      return;
    }
    noPoseFrames = 0;
    drawSkeleton(lm);
    const verdict = judge(lm);
    if (verdict) {
      call.textContent = verdict.isGood ? "✅ Good Depth" : "❌ No Lift";
      call.className = verdict.isGood ? "good" : "fail";
      detail.textContent = `Hip–Knee: ${verdict.marginCm.toFixed(1)} cm`;
      score.textContent = `Depth: ${(verdict.depthPct*100).toFixed(0)} %`;
    }
    debug.textContent = `Frames: ${frameCount}`;
  });

  // Real-time source: either camera stream or video file
  async function cameraLoop() {
    // Camera helper isn't used to avoid orientation mismatch; we send frames manually
    if (video.readyState >= 2) {
      await pose.send({image: video});
    }
    if (!processingVideoFile) requestAnimationFrame(cameraLoop);
  }

  // For video files, we run pose on each animation frame while playing
  function videoLoop() {
    if (!processingVideoFile) return;
    if (video.paused || video.ended) return;
    pose.send({image: video}).then(()=>{
      requestAnimationFrame(videoLoop);
    });
  }

  // Start loops
  if (!processingVideoFile) {
    cameraLoop();
  }

  video.addEventListener('play', ()=>{
    setCanvasToVideo();
    if (processingVideoFile) requestAnimationFrame(videoLoop);
  });

  // Restart pose when accuracy level changes
  accSel.addEventListener('change', ()=>{
    setupPose();
  }, {once:true});
}

async function main() {
  await listCameras();
  await startCamera();
  await setupPose();
}
main().catch(e=>{
  debug.textContent = "Error: " + e.message;
});
