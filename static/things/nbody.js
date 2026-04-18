// Ideal gas in a 2D box, integrated with velocity Verlet on the GPU.
//
// Integrator (two compute passes per step, holding a persistent acceleration
// buffer between steps):
//
//   Pass 1  (kick-drift):
//     v_{n+1/2} = v_n     + (dt/2) * a_n
//     x_{n+1}   = x_n     +  dt    * v_{n+1/2}
//
//   Pass 2  (force + final-kick):
//     a_{n+1}   = F(x_{n+1}) / m       (direct O(N^2) summation)
//     v_{n+1}   = v_{n+1/2} + (dt/2) * a_{n+1}
//
// Soft-sphere repulsion U(r) = (1/2) k (d - r)^2 for r < d; reflecting walls.
// Initial conditions: all particles at speed V0 with random direction, so
// the initial speed PDF is f_0(v) = delta(v - V0). Elastic collisions drive
// the system toward the 2D Maxwell-Boltzmann distribution.
//
// Particle colour is a piecewise-linear lookup into a thermal palette
// (pale blue-white at v = 0, navy at ~v_mp/2, turquoise, teal, yellow,
// orange, through to deep red at high speed), sampled at the vertex stage
// so each particle is shaded uniformly.

const N              = 4096;
const WORKGROUP_SIZE = 64;
const DT_BASE        = 0.0002;
const L              = 2.0;
const DIAMETER       = 0.02;
const STIFFNESS      = 2000.0;
const V0             = 0.5;
const V_COLOR_MAX    = 1.4;   // speeds >= this saturate to the hottest colour

// 10-stop thermal palette sampled from the reference weather chart.
// Indexed cold (t=0) -> hot (t=1).
const PALETTE = [
  [0.91, 0.93, 0.96],  // barely-there blue/white (coldest)
  [0.72, 0.77, 0.86],  // light gray-blue
  [0.43, 0.53, 0.69],  // medium blue
  [0.18, 0.27, 0.47],  // navy (freezing dividing wall)
  [0.21, 0.52, 0.60],  // turquoise
  [0.51, 0.65, 0.47],  // lime-teal
  [0.79, 0.65, 0.31],  // mellow yellow
  [0.78, 0.49, 0.23],  // warm orange/gold
  [0.70, 0.22, 0.37],  // scorching pink/red
  [0.29, 0.10, 0.16],  // very hot dark maroon
];

// --- DOM refs -------------------------------------------------------------

const canvas      = document.getElementById('sim-canvas');
const errorEl     = document.getElementById('sim-error');
const pauseBtn    = document.getElementById('sim-pause');
const resetBtn    = document.getElementById('sim-reset');
const speedSlider = document.getElementById('sim-speed');
const speedLabel  = document.getElementById('sim-speed-value');
const fpsEl       = document.getElementById('sim-fps');

function fail(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
  canvas.style.display = 'none';
}

if (!navigator.gpu) {
  fail('WebGPU is not available in this browser. Try Chrome/Edge 113+ or Safari 18+.');
  throw new Error('no webgpu');
}

// --- Histogram canvas (created dynamically) -------------------------------

const histWrap = document.createElement('div');
histWrap.className = 'sim-histogram';
histWrap.innerHTML = `<canvas id="hist-canvas" width="720" height="220"></canvas>`;
canvas.parentElement.appendChild(histWrap);
const histCanvas = document.getElementById('hist-canvas');
const histCtx    = histCanvas.getContext('2d');

function paletteJS(t) {
  const tc  = Math.max(0, Math.min(1, t));
  const idx = tc * (PALETTE.length - 1);
  const i0  = Math.floor(idx);
  const i1  = Math.min(i0 + 1, PALETTE.length - 1);
  const a   = idx - i0;
  const r = PALETTE[i0][0] * (1 - a) + PALETTE[i1][0] * a;
  const g = PALETTE[i0][1] * (1 - a) + PALETTE[i1][1] * a;
  const b = PALETTE[i0][2] * (1 - a) + PALETTE[i1][2] * a;
  return `rgb(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(b*255)})`;
}

// --- WebGPU setup ---------------------------------------------------------

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) { fail('No suitable GPU adapter found.'); throw new Error('no adapter'); }
const device  = await adapter.requestDevice();
const context = canvas.getContext('webgpu');
const format  = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'premultiplied' });

function resizeCanvas() {
  const dpr  = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  canvas.width  = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const hrect = histCanvas.getBoundingClientRect();
  histCanvas.width  = Math.floor(hrect.width * dpr);
  histCanvas.height = Math.floor(hrect.height * dpr);
  histCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// --- Initial conditions ---------------------------------------------------

function initialConditions() {
  const data = new Float32Array(N * 4);
  const margin = DIAMETER * 1.5;
  for (let i = 0; i < N; i++) {
    data[i * 4 + 0] = (Math.random() - 0.5) * (L - 2 * margin);
    data[i * 4 + 1] = (Math.random() - 0.5) * (L - 2 * margin);
    const theta = Math.random() * 2 * Math.PI;
    data[i * 4 + 2] = V0 * Math.cos(theta);
    data[i * 4 + 3] = V0 * Math.sin(theta);
  }
  return data;
}

const particleByteSize = N * 4 * 4;      // 16 bytes/particle: pos (vec2), vel (vec2)
const accByteSize      = N * 2 * 4;      //  8 bytes/particle: acc (vec2)

const bufferA = device.createBuffer({
  size: particleByteSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const bufferB = device.createBuffer({
  size: particleByteSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const accBuffer = device.createBuffer({
  size: accByteSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

function uploadInitial() {
  const data = initialConditions();
  device.queue.writeBuffer(bufferA, 0, data);
  device.queue.writeBuffer(bufferB, 0, data);
  // Zero the acceleration buffer. Since particles don't overlap at t=0, the
  // true initial accelerations are zero anyway.
  device.queue.writeBuffer(accBuffer, 0, new Float32Array(N * 2));
}
uploadInitial();

// Params uniform: [dt, diameter, stiffness, half_L].
const paramsBuffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
let speed = parseFloat(speedSlider.value);
function writeParams() {
  device.queue.writeBuffer(paramsBuffer, 0,
    new Float32Array([DT_BASE * speed, DIAMETER, STIFFNESS, L * 0.5]));
}
writeParams();

// --- Shared WGSL snippets -------------------------------------------------

const WGSL_PARTICLE = /* wgsl */`
  struct Particle { pos: vec2<f32>, vel: vec2<f32> };
  struct Params { dt: f32, diameter: f32, stiffness: f32, half_L: f32 };
`;

// --- Pass 1: kick-drift ---------------------------------------------------
//
// Reads:  src (pos, vel), acc[]
// Writes: dst (new pos, v_half)

const module1 = device.createShaderModule({ code: /* wgsl */`
  ${WGSL_PARTICLE}
  @group(0) @binding(0) var<storage, read>       src: array<Particle>;
  @group(0) @binding(1) var<storage, read_write> dst: array<Particle>;
  @group(0) @binding(2) var<storage, read>       acc: array<vec2<f32>>;
  @group(0) @binding(3) var<uniform>             params: Params;

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&src);
    if (i >= n) { return; }

    let p  = src[i];
    let a  = acc[i];
    let dt = params.dt;

    var v_half = p.vel + 0.5 * a * dt;
    var x_new  = p.pos + v_half * dt;

    // Reflecting walls: mirror position and flip the normal velocity.
    let hL = params.half_L;
    if (x_new.x >  hL) { x_new.x =  2.0 * hL - x_new.x; v_half.x = -v_half.x; }
    if (x_new.x < -hL) { x_new.x = -2.0 * hL - x_new.x; v_half.x = -v_half.x; }
    if (x_new.y >  hL) { x_new.y =  2.0 * hL - x_new.y; v_half.y = -v_half.y; }
    if (x_new.y < -hL) { x_new.y = -2.0 * hL - x_new.y; v_half.y = -v_half.y; }

    dst[i].pos = x_new;
    dst[i].vel = v_half;
  }
`});

const layout1 = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ],
});
const pipeline1 = device.createComputePipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [layout1] }),
  compute: { module: module1, entryPoint: 'main' },
});
function makeP1Bind(src, dst) {
  return device.createBindGroup({
    layout: layout1,
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: accBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ],
  });
}
const p1BindAB = makeP1Bind(bufferA, bufferB);   // src=A, dst=B
const p1BindBA = makeP1Bind(bufferB, bufferA);   // src=B, dst=A

// --- Pass 2: force computation + final kick -------------------------------
//
// Reads/writes state in place: reads all positions, reads own v_half, writes
// own new velocity. Also writes the new acceleration to acc[].

const module2 = device.createShaderModule({ code: /* wgsl */`
  ${WGSL_PARTICLE}
  @group(0) @binding(0) var<storage, read_write> state: array<Particle>;
  @group(0) @binding(1) var<storage, read_write> acc:   array<vec2<f32>>;
  @group(0) @binding(2) var<uniform>             params: Params;

  var<workgroup> tile: array<vec2<f32>, ${WORKGROUP_SIZE}>;

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn main(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id)  lid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&state);

    // Own position (sentinel used for any out-of-range lanes so they still
    // participate in workgroupBarrier uniformly).
    var my_pos: vec2<f32>;
    if (i < n) { my_pos = state[i].pos; }
    else       { my_pos = vec2<f32>(1e30, 1e30); }

    let d  = params.diameter;
    let d2 = d * d;
    var a_new = vec2<f32>(0.0, 0.0);

    let tiles = (n + ${WORKGROUP_SIZE}u - 1u) / ${WORKGROUP_SIZE}u;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
      let j = t * ${WORKGROUP_SIZE}u + lid.x;
      if (j < n) { tile[lid.x] = state[j].pos; }
      else       { tile[lid.x] = vec2<f32>(1e30, 1e30); }
      workgroupBarrier();

      for (var k: u32 = 0u; k < ${WORKGROUP_SIZE}u; k = k + 1u) {
        let dr = my_pos - tile[k];
        let r2 = dot(dr, dr);
        if (r2 < d2 && r2 > 1e-12) {
          let r = sqrt(r2);
          let f = params.stiffness * (d - r) / r;
          a_new = a_new + dr * f;
        }
      }
      workgroupBarrier();
    }

    if (i >= n) { return; }
    let v_half = state[i].vel;
    state[i].vel = v_half + 0.5 * a_new * params.dt;
    acc[i] = a_new;
  }
`});

const layout2 = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ],
});
const pipeline2 = device.createComputePipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [layout2] }),
  compute: { module: module2, entryPoint: 'main' },
});
function makeP2Bind(state) {
  return device.createBindGroup({
    layout: layout2,
    entries: [
      { binding: 0, resource: { buffer: state } },
      { binding: 1, resource: { buffer: accBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });
}
const p2BindA = makeP2Bind(bufferA);
const p2BindB = makeP2Bind(bufferB);

// --- Render pipeline ------------------------------------------------------
//
// WGSL palette: the JS array is baked into the shader as a const. The speed
// -> t mapping is computed in the vertex stage; the fragment stage just
// alpha-masks a disk.

function paletteWGSL() {
  // Emit "vec3<f32>(r, g, b)," lines
  return PALETTE.map(c =>
    `    vec3<f32>(${c[0].toFixed(3)}, ${c[1].toFixed(3)}, ${c[2].toFixed(3)}),`
  ).join('\n');
}

const moduleRender = device.createShaderModule({ code: /* wgsl */`
  ${WGSL_PARTICLE}
  @group(0) @binding(0) var<storage, read> particles: array<Particle>;

  fn palette(t: f32) -> vec3<f32> {
    let stops = array<vec3<f32>, ${PALETTE.length}>(
${paletteWGSL()}
    );
    let tc  = clamp(t, 0.0, 1.0);
    let idx = tc * ${(PALETTE.length - 1).toFixed(1)};
    let i0  = u32(floor(idx));
    let i1  = min(i0 + 1u, ${PALETTE.length - 1}u);
    let a   = idx - f32(i0);
    return mix(stops[i0], stops[i1], a);
  }

  struct VSOut {
    @builtin(position) pos:   vec4<f32>,
    @location(0)       uv:    vec2<f32>,
    @location(1)       color: vec3<f32>,
  };

  @vertex
  fn vs(@builtin(vertex_index)   vi: u32,
        @builtin(instance_index) ii: u32) -> VSOut {
    var corners = array<vec2<f32>, 6>(
      vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
      vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    );
    let c    = corners[vi];
    let p    = particles[ii];
    // Visual radius slightly larger than collision radius so the palette is
    // legible at this N.
    let size = ${(DIAMETER * 0.3).toFixed(5)};
    let t    = clamp(length(p.vel) / ${V_COLOR_MAX}, 0.0, 1.0);
    var out: VSOut;
    out.pos   = vec4<f32>(p.pos + c * size, 0.0, 1.0);
    out.uv    = c;
    out.color = palette(t);
    return out;
  }

  @fragment
  fn fs(in: VSOut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    if (r > 1.0) { discard; }
    // Crisp disk with a 1-px antialiased edge.
    let alpha = smoothstep(1.0, 0.85, r);
    return vec4<f32>(in.color * alpha, alpha);   // premultiplied alpha
  }
`});

const layoutR = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
  ],
});
const pipelineR = device.createRenderPipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [layoutR] }),
  vertex:   { module: moduleRender, entryPoint: 'vs' },
  fragment: { module: moduleRender, entryPoint: 'fs', targets: [{
    format,
    blend: {
      color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    },
  }] },
  primitive: { topology: 'triangle-list' },
});
const renderBindA = device.createBindGroup({
  layout: layoutR,
  entries: [{ binding: 0, resource: { buffer: bufferA } }],
});
const renderBindB = device.createBindGroup({
  layout: layoutR,
  entries: [{ binding: 0, resource: { buffer: bufferB } }],
});

// --- Readback for histogram ----------------------------------------------

const readbackBuffer = device.createBuffer({
  size: particleByteSize,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
let readbackInFlight = false;

const NUM_BINS = 32;
const V_MAX    = 1.4;
let histCounts = new Float32Array(NUM_BINS);
let kToverM    = 0.5 * V0 * V0;

function startReadback() {
  if (readbackInFlight) return;
  readbackInFlight = true;
  const src = latestIsA ? bufferA : bufferB;
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, readbackBuffer, 0, particleByteSize);
  device.queue.submit([encoder.finish()]);
  readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
    const data = new Float32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();
    processReadback(data);
    readbackInFlight = false;
  }).catch(() => { readbackInFlight = false; });
}

function processReadback(data) {
  const counts = new Float32Array(NUM_BINS);
  const binW   = V_MAX / NUM_BINS;
  let v2sum = 0;
  for (let i = 0; i < N; i++) {
    const vx = data[i * 4 + 2];
    const vy = data[i * 4 + 3];
    const v2 = vx * vx + vy * vy;
    v2sum += v2;
    const v  = Math.sqrt(v2);
    const b  = Math.min(Math.floor(v / binW), NUM_BINS - 1);
    counts[b] += 1;
  }
  const alpha = 0.25;
  for (let i = 0; i < NUM_BINS; i++) {
    histCounts[i] = (1 - alpha) * histCounts[i] + alpha * counts[i];
  }
  kToverM = v2sum / (2 * N);
}

function drawHistogram() {
  const W = histCanvas.clientWidth;
  const H = histCanvas.clientHeight;

  // Dark background so pale (cold) bars remain visible.
  histCtx.fillStyle = '#0b0d14';
  histCtx.fillRect(0, 0, W, H);

  const padL = 44, padR = 16, padT = 20, padB = 36;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const binW  = V_MAX / NUM_BINS;

  const density = new Float32Array(NUM_BINS);
  for (let i = 0; i < NUM_BINS; i++) density[i] = histCounts[i] / (N * binW);

  const vMp         = Math.sqrt(kToverM);
  const fPeakTheory = vMp / kToverM * Math.exp(-0.5);
  const yMax        = Math.max(fPeakTheory * 1.4, 1.0);

  // Bars coloured by the palette at each bin's centre speed.
  for (let i = 0; i < NUM_BINS; i++) {
    const vCenter = (i + 0.5) * binW;
    histCtx.fillStyle = paletteJS(vCenter / V_COLOR_MAX);
    const x  = padL + (i / NUM_BINS) * plotW;
    const bw = plotW / NUM_BINS - 1;
    const bh = (density[i] / yMax) * plotH;
    histCtx.fillRect(x, padT + plotH - bh, bw, bh);
  }

  // Theoretical MB curve (2D): f(v) = (v / s^2) exp(-v^2 / 2 s^2).
  histCtx.strokeStyle = '#ff5b6f';
  histCtx.lineWidth = 2;
  histCtx.beginPath();
  const s2 = kToverM;
  for (let i = 0; i <= 200; i++) {
    const v = (i / 200) * V_MAX;
    const f = (v / s2) * Math.exp(-v * v / (2 * s2));
    const x = padL + (v / V_MAX) * plotW;
    const y = padT + plotH - (f / yMax) * plotH;
    if (i === 0) histCtx.moveTo(x, y);
    else         histCtx.lineTo(x, y);
  }
  histCtx.stroke();

  // Axes
  histCtx.strokeStyle = '#555a6a';
  histCtx.lineWidth   = 1;
  histCtx.beginPath();
  histCtx.moveTo(padL, padT);
  histCtx.lineTo(padL, padT + plotH);
  histCtx.lineTo(padL + plotW, padT + plotH);
  histCtx.stroke();

  // Ticks and labels
  histCtx.fillStyle = '#bac0cf';
  histCtx.font      = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  histCtx.textAlign = 'center';
  for (let tv = 0; tv <= V_MAX + 1e-6; tv += 0.2) {
    const x = padL + (tv / V_MAX) * plotW;
    histCtx.beginPath();
    histCtx.moveTo(x, padT + plotH);
    histCtx.lineTo(x, padT + plotH + 4);
    histCtx.stroke();
    histCtx.fillText(tv.toFixed(1), x, padT + plotH + 16);
  }
  histCtx.textAlign = 'left';
  histCtx.fillText('speed v', padL + plotW - 50, padT + plotH + 30);
  histCtx.fillText('f(v)',    6, padT + 4);

  // Legend
  histCtx.fillStyle = paletteJS(0.35);
  histCtx.fillRect(padL + 6, padT + 4, 10, 10);
  histCtx.fillStyle = '#d8dde8';
  histCtx.fillText('measured', padL + 22, padT + 13);
  histCtx.strokeStyle = '#ff5b6f';
  histCtx.beginPath();
  histCtx.moveTo(padL + 100, padT + 9);
  histCtx.lineTo(padL + 114, padT + 9);
  histCtx.stroke();
  histCtx.fillText('Maxwell-Boltzmann', padL + 120, padT + 13);

  histCtx.textAlign = 'right';
  histCtx.fillText(`kT/m = ${kToverM.toFixed(4)}`, W - padR, padT + 13);
}

// --- Main loop ------------------------------------------------------------

let paused    = false;
let latestIsA = true;       // which particle buffer holds the latest state
let lastTime  = performance.now();
let fpsAcc = 0, fpsCount = 0;
let frameIdx = 0;

function recordVVStep(encoder) {
  // Pass 1: kick-drift src -> dst.
  {
    const p1 = encoder.beginComputePass();
    p1.setPipeline(pipeline1);
    p1.setBindGroup(0, latestIsA ? p1BindAB : p1BindBA);
    p1.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
    p1.end();
  }
  // After pass 1, the freshly-written buffer is the "other" one; flip.
  latestIsA = !latestIsA;

  // Pass 2: force + final kick, operating on the latest buffer in place.
  {
    const p2 = encoder.beginComputePass();
    p2.setPipeline(pipeline2);
    p2.setBindGroup(0, latestIsA ? p2BindA : p2BindB);
    p2.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
    p2.end();
  }
}

function recordRender(encoder) {
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.031, g: 0.039, b: 0.078, a: 1.0 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
  });
  pass.setPipeline(pipelineR);
  pass.setBindGroup(0, latestIsA ? renderBindA : renderBindB);
  pass.draw(6, N);
  pass.end();
}

function loop() {
  const now = performance.now();
  const dt  = now - lastTime;
  lastTime  = now;
  fpsAcc   += 1000 / dt; fpsCount++;
  if (fpsCount >= 30) {
    fpsEl.textContent = `${(fpsAcc / fpsCount).toFixed(0)} fps · N = ${N}`;
    fpsAcc = 0; fpsCount = 0;
  }

  const encoder = device.createCommandEncoder();
  if (!paused) recordVVStep(encoder);
  recordRender(encoder);
  device.queue.submit([encoder.finish()]);

  if (frameIdx % 10 === 0) startReadback();
  drawHistogram();

  frameIdx++;
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// --- Controls -------------------------------------------------------------

pauseBtn.addEventListener('click', () => {
  paused = !paused;
  pauseBtn.textContent = paused ? 'Resume' : 'Pause';
});
resetBtn.addEventListener('click', () => {
  uploadInitial();           // also zeros accBuffer
  latestIsA  = true;
  histCounts = new Float32Array(NUM_BINS);
  kToverM    = 0.5 * V0 * V0;
});
speedSlider.addEventListener('input', () => {
  speed = parseFloat(speedSlider.value);
  speedLabel.textContent = `${speed.toFixed(2)}×`;
  writeParams();
});
