// Robo Judge v6: back camera default, robust video init, debug messages
const video = document.getElementById('video');
const stage = document.getElementById('stage');
const ctx = stage.getContext('2d');
const tol=document.getElementById('tol'), tolVal=document.getElementById('tolVal');
const smooth=document.getElementById('smooth'), smVal=document.getElementById('smVal');
const cameraSel=document.getElementById('camera'), flipBtn=document.getElementById('flip');
const accSel=document.getElementById('accuracy'), fileInput=document.getElementById('file');
const depthStatus=document.getElementById('depthStatus'), marginEl=document.getElementById('margin');
const kneeAnglesEl=document.getElementById('kneeAngles'), hipAnglesEl=document.getElementById('hipAngles');
const statusEl=document.getElementById('status'), frameCountEl=document.getElementById('frameCount');

let currentDeviceId=null, useFront=false, processingVideoFile=false;
let frameCount=0; let cmPerPx=0.2;

function setStageSize(){
  if(video.videoWidth && video.videoHeight){
    stage.width=video.videoWidth; stage.height=video.videoHeight;
    stage.style.width="100%"; stage.style.height="100%";
  }
}
function logStatus(msg){ statusEl.textContent=msg; }
function pushHist(hist,v,max){ hist.push(v); if(hist.length>max) hist.shift(); }
function avg(arr){ return arr.reduce((s,v)=>s+v,0)/Math.max(arr.length,1); }

function judgeDepth(lm){
  const lh=lm[23], rh=lm[24], lk=lm[25], rk=lm[26];
  if(!lh||!rh||!lk||!rk) return null;
  const hipY=Math.min(lh.y,rh.y), kneeY=Math.max(lk.y,rk.y);
  const tolY=+tol.value;
  const isGood=(hipY <= kneeY+tolY);
  const marginNorm=(kneeY-hipY);
  const marginPx=marginNorm*stage.height;
  const marginCm=marginPx*cmPerPx;
  return {isGood, marginCm};
}
function angleDeg(a,b,c){ if(!a||!b||!c) return null;
  const ABx=a.x-b.x,ABy=a.y-b.y, CBx=c.x-b.x,CBy=c.y-b.y;
  const dot=ABx*CBx+ABy*CBy;
  const mag1=Math.hypot(ABx,ABy), mag2=Math.hypot(CBx,CBy);
  if(!mag1||!mag2) return null;
  return Math.round(Math.acos(dot/(mag1*mag2))*180/Math.PI);
}
function computeAngles(lm){
  const Lk=angleDeg(lm[23],lm[25],lm[27]), Rk=angleDeg(lm[24],lm[26],lm[28]);
  const Lh=angleDeg(lm[11],lm[23],lm[25]), Rh=angleDeg(lm[12],lm[24],lm[26]);
  kneeAnglesEl.textContent=`${Lk??"—"}° / ${Rk??"—"}°`;
  hipAnglesEl.textContent=`${Lh??"—"}° / ${Rh??"—"}°`;
}

function drawSkeleton(lm,isGood){
  if(!lm) return;
  ctx.lineWidth=3; ctx.strokeStyle=isGood?"rgba(90,245,118,0.9)":"rgba(255,107,107,0.95)";
  const px=p=>({x:p.x*stage.width,y:p.y*stage.height});
  const pairs=[[11,13],[13,15],[12,14],[14,16],[23,25],[25,27],[24,26],[26,28],[11,12],[23,24],[11,23],[12,24]];
  pairs.forEach(([a,b])=>{const pa=lm[a],pb=lm[b];if(!pa||!pb)return;const A=px(pa),B=px(pb);ctx.beginPath();ctx.moveTo(A.x,A.y);ctx.lineTo(B.x,B.y);ctx.stroke();});
  ctx.fillStyle="#fff"; lm.forEach(p=>{if(!p)return;const P=px(p);ctx.beginPath();ctx.arc(P.x,P.y,3,0,Math.PI*2);ctx.fill();});
}

async function listCameras(){
  const devices=await navigator.mediaDevices.enumerateDevices();
  cameraSel.innerHTML='';
  const vids=devices.filter(d=>d.kind==='videoinput');
  vids.forEach((d,i)=>{const opt=document.createElement('option');opt.value=d.deviceId;opt.textContent=d.label||`Camera ${i+1}`;cameraSel.appendChild(opt);});
  if(vids.length&&!currentDeviceId){
    // Default to back camera if available
    const back=vids.find(d=>/back|rear/i.test(d.label));
    currentDeviceId=back?back.deviceId:vids[0].deviceId;
  }
  if(currentDeviceId) cameraSel.value=currentDeviceId;
}

async function startCamera(){
  processingVideoFile=false;
  logStatus("Requesting camera…");
  try{
    const constraints={video: currentDeviceId?{deviceId:{exact:currentDeviceId}}:{facingMode:"environment"}, audio:false};
    const stream=await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject=stream; video.muted=true;
    await video.play();
    video.addEventListener('loadedmetadata',()=>{setStageSize();logStatus("Camera active");});
  }catch(e){ logStatus("Camera error: "+e.message); }
}

fileInput.addEventListener('change',async e=>{
  const file=e.target.files[0]; if(!file)return;
  processingVideoFile=true;
  if(video.srcObject){video.srcObject.getTracks().forEach(t=>t.stop());video.srcObject=null;}
  video.src=URL.createObjectURL(file); video.muted=true;
  try{await video.play(); logStatus("Video file playing");}catch(err){logStatus("Video play error: "+err.message);}
  video.addEventListener('loadedmetadata',()=>{setStageSize();});
});

async function main(){
  await listCameras();
  await startCamera();
  const pose=new Pose.Pose({locateFile:(f)=>`https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${f}`});
  pose.setOptions({modelComplexity:+accSel.value,smoothLandmarks:true,minDetectionConfidence:0.5,minTrackingConfidence:0.5});
  pose.onResults(res=>{
    frameCount++; frameCountEl.textContent=frameCount;
    setStageSize();
    ctx.clearRect(0,0,stage.width,stage.height); ctx.drawImage(video,0,0,stage.width,stage.height);
    if(!res.poseLandmarks){depthStatus.textContent="No pose";return;}
    const verdict=judgeDepth(res.poseLandmarks);
    drawSkeleton(res.poseLandmarks,verdict?.isGood);
    computeAngles(res.poseLandmarks);
    if(verdict){depthStatus.textContent=verdict.isGood?"GOOD DEPTH":"NOT DEEP ENOUGH";marginEl.textContent=`${verdict.marginCm.toFixed(1)} cm`;}
  });
  function loop(){ pose.send({image:video}).catch(e=>logStatus("Pose error:"+e.message)).finally(()=>requestAnimationFrame(loop)); }
  loop();
}
main().catch(e=>logStatus("Init error: "+e.message));

tol.addEventListener('input',()=>tolVal.textContent=(+tol.value).toFixed(3));
smooth.addEventListener('input',()=>smVal.textContent=smooth.value);
cameraSel.addEventListener('change',()=>{currentDeviceId=cameraSel.value;startCamera();});
flipBtn.addEventListener('click',()=>{useFront=!useFront;currentDeviceId=null;startCamera();});
accSel.addEventListener('change',()=>{});
