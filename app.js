// SquatJudge Web - MediaPipe Pose implementation
// Depth rule: hip crease below top of knee (with tolerance)
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const call = document.getElementById('call');
const detail = document.getElementById('detail');
const score = document.getElementById('score');
const tol = document.getElementById('tol');
const tolVal = document.getElementById('tolVal');
const smooth = document.getElementById('smooth');
const smVal = document.getElementById('smVal');
const cameraSel = document.getElementById('camera');
const flipBtn = document.getElementById('flip');

let currentDeviceId = null;
let useFront = false;

function resize() {
  const r = document.getElementById('video-wrap').getBoundingClientRect();
  canvas.width = r.width; canvas.height = r.height;
}
window.addEventListener('resize', resize);

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
  await startCamera();
});

flipBtn.addEventListener('click', async ()=>{
  useFront = !useFront;
  await startCamera();
});

async function startCamera() {
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
  resize();
}

function lerp(a,b,t){return a+(b-a)*t}

const hipHist = [], kneeHist = [];
function pushHist(hist, v, max){ hist.push(v); if(hist.length>max) hist.shift(); }
function avg(arr){ return arr.reduce((s,v)=>s+v,0)/arr.length; }

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
}

function judge(landmarks) {
  const lh = landmarks[23], rh = landmarks[24];
  const lk = landmarks[25], rk = landmarks[26];
  if(!lh||!rh||!lk||!rk) return null;

  // MediaPipe coordinates origin top-left, y increases downward.
  const hipY = Math.min(lh.y, rh.y);     // lower hip (closer to ground)
  const kneeY = Math.max(lk.y, rk.y);    // top of knees
  pushHist(hipHist, hipY, +smooth.value);
  pushHist(kneeHist, kneeY, +smooth.value);
  const hipYsm = avg(hipHist), kneeYsm = avg(kneeHist);

  const tolY = +tol.value; // normalized units
  const isGood = (hipYsm <= kneeYsm + tolY);

  const marginNorm = (kneeYsm - hipYsm); // positive = hip below
  const cmPerPx = estimateCmPerPx(landmarks);
  const marginPx = marginNorm * canvas.height;
  const marginCm = marginPx * cmPerPx;
  const depthPct = Math.max(0, Math.min(1.2, marginCm / 4.0));

  return {isGood, marginCm, depthPct};
}

async function main() {
  await listCameras();
  await startCamera();

  const pose = new Pose.Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`
  });
  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  const camera = new Camera(video, {
    onFrame: async () => {
      await pose.send({image: video});
    },
    width: 720,
    height: 1280
  });
  camera.start();

  pose.onResults((res)=>{
    resize();
    const lm = res.poseLandmarks;
    drawSkeleton(lm);
    const verdict = lm ? judge(lm) : null;
    if (verdict) {
      call.textContent = verdict.isGood ? "✅ Good Depth" : "❌ No Lift";
      call.className = verdict.isGood ? "good" : "fail";
      detail.textContent = `Hip–Knee: ${verdict.marginCm.toFixed(1)} cm`;
      score.textContent = `Depth: ${(verdict.depthPct*100).toFixed(0)} %`;
    } else {
      call.textContent = "Align side view • Stand tall";
      call.className = "";
      detail.textContent = "Hip–Knee: -- cm";
      score.textContent = "Depth: -- %";
    }
  });
}

main();
