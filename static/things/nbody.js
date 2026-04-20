// Ideal gas in a 2D box with hard-sphere elastic collisions, on the GPU.
//
// Integrator (two compute passes per step; no forces, no persistent state):
//
//   Pass 1  (drift + reflecting walls):
//     x_{n+1} = x_n + v_n * dt
//     mirror position and flip normal velocity on wall contact
//
//   Pass 2  (binary elastic collisions, direct O(N^2) scan):
//     for each j with r := |x_i - x_j| < d  and  (v_i - v_j).(x_i - x_j) < 0:
//       v_i <- v_i - ((v_i - v_j).(x_i - x_j) / r^2) (x_i - x_j)
//   Each GPU lane updates only its own velocity, computing the impulse
//   symmetrically from the pre-collision state of every overlapping,
//   approaching neighbour. No cross-lane writes; pair-wise momentum and
//   kinetic energy are conserved exactly.
//
// The "Speed" slider controls how many integrator steps are run between
// rendered frames. dt is fixed at DT_BASE. "10x" means 10 dt of simulated
// time are advanced per displayed frame. Negative values flip the sign
// of dt (and of the approaching gate); time-stepped hard-sphere dynamics
// is reversible only up to the finite-dt error in collision timing, so
// reverse play retraces the forward trajectory approximately rather than
// to round-off as velocity Verlet did.
//
// Hard-sphere diameter d; reflecting walls. Initial conditions: all
// particles at speed V0 with random direction, so the initial speed PDF
// is f_0(v) = delta(v - V0). Elastic binary collisions drive the system
// toward the 2D Maxwell-Boltzmann distribution.
//
// Particle colour is a piecewise-linear lookup into a thermal palette
// (pale blue-white at v = 0, navy at ~v_mp/2, turquoise, teal, yellow,
// orange, through to deep red at high speed), sampled at the vertex stage
// so each particle is shaded uniformly.

const N              = 4096;
const WORKGROUP_SIZE = 64;
const DT_BASE        = 0.001;
const L              = 2.0;
const DIAMETER       = 0.005;
const V0             = 1.0;
const V_COLOR_MAX    = 2.6;   // speeds >= this saturate to the hottest colour

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
  // Jittered square-lattice placement. With cell size l = L/M and
  // M = ceil(sqrt(N)), a symmetric jitter of span alpha*(l - d) with
  // alpha < 1 guarantees both pair separation > d and wall clearance
  // > d/2, so the kernel sees no overlaps at t = 0.
  const M      = Math.ceil(Math.sqrt(N));
  const cell   = L / M;
  const jitter = 0.9 * (cell - DIAMETER);
  const data   = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    const cx = i % M;
    const cy = Math.floor(i / M);
    const x0 = -L * 0.5 + (cx + 0.5) * cell;
    const y0 = -L * 0.5 + (cy + 0.5) * cell;
    data[i * 4 + 0] = x0 + (Math.random() - 0.5) * jitter;
    data[i * 4 + 1] = y0 + (Math.random() - 0.5) * jitter;
    const theta = Math.random() * 2 * Math.PI;
    // const theta = 1.5*Math.PI;
    data[i * 4 + 2] = V0 * Math.cos(theta);
    data[i * 4 + 3] = V0 * Math.sin(theta);
  }
  return data;
}

const particleByteSize = N * 4 * 4;      // 16 bytes/particle: pos (vec2), vel (vec2)

const bufferA = device.createBuffer({
  size: particleByteSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const bufferB = device.createBuffer({
  size: particleByteSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});

// Partner buffer. partnerBuffer[i] = j, the global index of i's chosen
// collision partner this substep, or -1 if no overlapping approaching
// neighbour exists. Written by Pass 2a, read by Pass 2b.
const partnerBuffer = device.createBuffer({
  size: N * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});

function uploadInitial() {
  const data = initialConditions();
  device.queue.writeBuffer(bufferA, 0, data);
  device.queue.writeBuffer(bufferB, 0, data);
}
uploadInitial();

// Params uniform: [dt, diameter, half_L]; padded to 16 B for UBO alignment.
const paramsBuffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// Slider state. `speed` is the signed slider value; `substeps` is the
// number of VV steps to run per rendered frame (|speed|); `dt` is DT_BASE
// with the sign of speed, so negative speed runs VV backward in time.
let speed    = parseFloat(speedSlider.value);
let substeps = Math.max(1, Math.abs(Math.round(speed)));
let dtSigned = Math.sign(speed || 1) * DT_BASE;

function writeParams() {
  device.queue.writeBuffer(paramsBuffer, 0,
    new Float32Array([dtSigned, DIAMETER, L * 0.5]));
}
writeParams();

// --- Shared WGSL snippets -------------------------------------------------

const WGSL_PARTICLE = /* wgsl */`
  struct Particle { pos: vec2<f32>, vel: vec2<f32> };
  struct Params { dt: f32, diameter: f32, half_L: f32 };
`;

// --- Pass 1: drift + reflecting walls ------------------------------------
//
// Reads:  src (pos, vel)
// Writes: dst (new pos, wall-corrected vel)

const module1 = device.createShaderModule({ code: /* wgsl */`
  ${WGSL_PARTICLE}
  @group(0) @binding(0) var<storage, read>       src: array<Particle>;
  @group(0) @binding(1) var<storage, read_write> dst: array<Particle>;
  @group(0) @binding(2) var<uniform>             params: Params;

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&src);
    if (i >= n) { return; }

    let p = src[i];
    var vel = p.vel;
    var pos = p.pos + vel * params.dt;

    // Reflecting walls: mirror position and flip the normal velocity.
    let hL = params.half_L;
    if (pos.x >  hL) { pos.x =  2.0 * hL - pos.x; vel.x = -vel.x; }
    if (pos.x < -hL) { pos.x = -2.0 * hL - pos.x; vel.x = -vel.x; }
    if (pos.y >  hL) { pos.y =  2.0 * hL - pos.y; vel.y = -vel.y; }
    if (pos.y < -hL) { pos.y = -2.0 * hL - pos.y; vel.y = -vel.y; }

    dst[i].pos = pos;
    dst[i].vel = vel;
  }
`});

// --- Bind-group layouts ---------------------------------------------------
//
// Pass 1  (drift):       src particles -> dst particles       (+ params)
// Pass 2a (find_partner): src particles -> partnerBuffer       (+ params)
// Pass 2b (apply):       src particles, partnerBuffer -> dst  (+ params)

const driftLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ],
});
const partnerLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ],
});
const collideLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
  ],
});

const pipeline1 = device.createComputePipeline({
  layout:  device.createPipelineLayout({ bindGroupLayouts: [driftLayout] }),
  compute: { module: module1, entryPoint: 'main' },
});

function makeDriftBind(src, dst) {
  return device.createBindGroup({
    layout: driftLayout,
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });
}
const driftAB = makeDriftBind(bufferA, bufferB);
const driftBA = makeDriftBind(bufferB, bufferA);

// --- Pass 2a: find each particle's mutual-best collision partner --------
//
// For each particle i, scan all others (tiled through workgroup shared
// memory) and record the index of the single most urgent overlapping,
// approaching neighbour — the one with the most negative vn*dt. Ties are
// broken by smallest global index. The resulting ranking is symmetric:
// for any pair (i, j) both lanes compute the same score and the same
// tie-break, so if (i, j) is i's best candidate among its approaches it
// is also j's best candidate among its approaches iff they mutually
// prefer each other.
//
// Reads:  src (post-drift pos, vel)
// Writes: partners[i] = j (i32) or -1

const modulePartner = device.createShaderModule({ code: /* wgsl */`
  ${WGSL_PARTICLE}
  @group(0) @binding(0) var<storage, read>       src:      array<Particle>;
  @group(0) @binding(1) var<storage, read_write> partners: array<i32>;
  @group(0) @binding(2) var<uniform>             params:   Params;

  var<workgroup> tile: array<Particle, ${WORKGROUP_SIZE}>;

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn main(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id)  lid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&src);

    var my_pos: vec2<f32>;
    var my_vel: vec2<f32>;
    if (i < n) { my_pos = src[i].pos; my_vel = src[i].vel; }
    else       { my_pos = vec2<f32>(1e30, 1e30); my_vel = vec2<f32>(0.0, 0.0); }

    let d  = params.diameter;
    let d2 = d * d;
    let dt = params.dt;

    var best_j:     i32 = -1;
    var best_score: f32 = 0.0;   // vn*dt; any approach has score < 0

    let tiles = (n + ${WORKGROUP_SIZE}u - 1u) / ${WORKGROUP_SIZE}u;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
      let base     = t * ${WORKGROUP_SIZE}u;
      let load_idx = base + lid.x;
      if (load_idx < n) { tile[lid.x] = src[load_idx]; }
      else              { tile[lid.x] = Particle(vec2<f32>(1e30, 1e30), vec2<f32>(0.0, 0.0)); }
      workgroupBarrier();

      for (var k: u32 = 0u; k < ${WORKGROUP_SIZE}u; k = k + 1u) {
        let j = base + k;
        if (j == i || j >= n) { continue; }
        let dr = my_pos - tile[k].pos;
        let r2 = dot(dr, dr);
        if (r2 < d2 && r2 > 1e-12) {
          let dv    = my_vel - tile[k].vel;
          let vn    = dot(dv, dr);
          let score = vn * dt;
          if (score < best_score) {
            best_score = score;
            best_j     = i32(j);
          } else if (score == best_score && i32(j) < best_j) {
            best_j = i32(j);
          }
        }
      }
      workgroupBarrier();
    }

    if (i < n) { partners[i] = best_j; }
  }
`});

const pipelinePartner = device.createComputePipeline({
  layout:  device.createPipelineLayout({ bindGroupLayouts: [partnerLayout] }),
  compute: { module: modulePartner, entryPoint: 'main' },
});

function makePartnerBind(src) {
  return device.createBindGroup({
    layout: partnerLayout,
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: partnerBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });
}
const partnerBindA = makePartnerBind(bufferA);
const partnerBindB = makePartnerBind(bufferB);

// --- Pass 2b: apply elastic impulse for mutual-best pairs ----------------
//
// For each particle i, read j = partners[i]. Fire the equal-mass elastic
// impulse only if partners[j] == i. Because the impulse rule is symmetric
// and both lanes compute J from the same pre-collision src state, the two
// sides of every fired pair apply exactly opposite impulses. Pairwise
// momentum and kinetic energy are conserved to machine precision; there
// is no sum-of-impulses cross term, so clusters of simultaneous overlaps
// no longer leak energy.
//
// Non-mutual overlaps are deferred: the `vn*dt < 0` gate persists while
// the pair is still approaching, so they will be resolved on a subsequent
// substep (typically the very next one).

const module2 = device.createShaderModule({ code: /* wgsl */`
  ${WGSL_PARTICLE}
  @group(0) @binding(0) var<storage, read>       src:      array<Particle>;
  @group(0) @binding(1) var<storage, read_write> dst:      array<Particle>;
  @group(0) @binding(2) var<uniform>             params:   Params;
  @group(0) @binding(3) var<storage, read>       partners: array<i32>;

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&src);
    if (i >= n) { return; }

    let my  = src[i];
    var vel = my.vel;

    let j_signed = partners[i];
    if (j_signed >= 0) {
      let j = u32(j_signed);
      // Mutual-best gate: fire only if j also chose i.
      if (partners[j] == i32(i)) {
        let other = src[j];
        let dr = my.pos - other.pos;
        let r2 = dot(dr, dr);
        if (r2 > 1e-12) {
          let dv = my.vel - other.vel;
          let vn = dot(dv, dr);
          vel = my.vel - (vn / r2) * dr;
        }
      }
    }

    dst[i].pos = my.pos;
    dst[i].vel = vel;
  }
`});

const pipeline2 = device.createComputePipeline({
  layout:  device.createPipelineLayout({ bindGroupLayouts: [collideLayout] }),
  compute: { module: module2, entryPoint: 'main' },
});

function makeCollideBind(src, dst) {
  return device.createBindGroup({
    layout: collideLayout,
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: paramsBuffer } },
      { binding: 3, resource: { buffer: partnerBuffer } },
    ],
  });
}
const collideAB = makeCollideBind(bufferA, bufferB);
const collideBA = makeCollideBind(bufferB, bufferA);

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
    let size = ${(DIAMETER * 1.0).toFixed(5)};
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

const NUM_BINS = 64;
const V_MAX    = 2.6;
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
    const v = Math.sqrt(v2);
    const b = Math.floor(v / binW);
    // Drop particles above the plot range rather than clamping them
    // into the last bin (which would spuriously inflate the tail).
    if (b < NUM_BINS) counts[b] += 1;
  }
  const alpha = 0.4;
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
    const bw = plotW / NUM_BINS - 0.5;
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

function recordStep(encoder) {
  // Pass 1: drift + walls, src -> dst.
  {
    const p1 = encoder.beginComputePass();
    p1.setPipeline(pipeline1);
    p1.setBindGroup(0, latestIsA ? driftAB : driftBA);
    p1.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
    p1.end();
  }
  latestIsA = !latestIsA;

  // Pass 2a: every particle picks its single most-urgent overlapping
  // approaching partner (or -1) and stores the index in partnerBuffer.
  {
    const pa = encoder.beginComputePass();
    pa.setPipeline(pipelinePartner);
    pa.setBindGroup(0, latestIsA ? partnerBindA : partnerBindB);
    pa.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
    pa.end();
  }

  // Pass 2b: apply the elastic impulse to pairs (i, j) that mutually chose
  // each other. Ping-pongs the particle buffer back so the caller's
  // latestIsA semantics are unchanged.
  {
    const p2 = encoder.beginComputePass();
    p2.setPipeline(pipeline2);
    p2.setBindGroup(0, latestIsA ? collideAB : collideBA);
    p2.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
    p2.end();
  }
  latestIsA = !latestIsA;
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
  if (!paused) {
    for (let s = 0; s < substeps; s++) recordStep(encoder);
  }
  recordRender(encoder);
  device.queue.submit([encoder.finish()]);

  // Kick a readback every frame; the in-flight guard naturally
  // rate-limits to the GPU round-trip (~30 Hz in practice).
  startReadback();
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
  uploadInitial();
  latestIsA  = true;
  histCounts = new Float32Array(NUM_BINS);
  kToverM    = 0.5 * V0 * V0;
});
speedSlider.addEventListener('input', () => {
  speed = parseFloat(speedSlider.value);
  // Integer substeps; 0 means no stepping (soft pause).
  substeps = Math.abs(Math.round(speed));
  // Signed dt for time reversal. Keep previous sign if speed is exactly 0
  // so that the integrator parameter is well-defined when stopped.
  if (speed !== 0) dtSigned = Math.sign(speed) * DT_BASE;
  speedLabel.textContent = `${Math.round(speed)}×`;
  writeParams();
});
