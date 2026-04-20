---
title: "Build a WebGPU ideal‑gas simulator with velocity Verlet"
date: 2026-04-19
math: true
---

This note walks through building a 2D ideal‑gas simulation that runs entirely on the GPU using **WebGPU**. Particles interact via soft‑sphere repulsion inside a box, and their speeds are visualized with a temperature‑mapped colour palette. A live histogram compares the measured speed distribution to the theoretical Maxwell–Boltzmann curve.

We’ll use **velocity Verlet** integration – a symplectic, time‑reversible method that keeps energy stable even with stiff collisions. The whole simulation (4096 particles, ~16 million pair interactions per step) runs comfortably at 60 fps in a browser.

---

## 1. Prerequisites

- A browser with WebGPU enabled (Chrome/Edge 113+, Safari 18+).
- Basic familiarity with JavaScript and the WebGPU API.
- A local web server (or Hugo) to serve the files – WebGPU requires a secure context (`https` or `localhost`).

---

## 2. Project skeleton

Create an `index.html` file with a `<canvas>` for the simulation, a few control buttons, and a `<div>` that will hold the histogram.

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { background: #0b0d14; color: #e0e5f0; font-family: system-ui; }
    #sim-canvas { width: 800px; height: 600px; display: block; }
    .controls { margin: 12px 0; }
    .sim-histogram canvas { width: 720px; height: 220px; background: #0b0d14; }
  </style>
</head>
<body>
  <canvas id="sim-canvas" width="800" height="600"></canvas>
  <div class="controls">
    <button id="sim-pause">Pause</button>
    <button id="sim-reset">Reset</button>
    <input type="range" id="sim-speed" min="-10" max="10" value="1" step="1">
    <span id="sim-speed-value">1×</span>
    <span id="sim-fps"></span>
  </div>
  <div id="sim-error" hidden style="color:#f88;"></div>
  <script type="module" src="sim.js"></script>
</body>
</html>
```

All WebGPU code will go into sim.js. We’ll build it step by step.

---

## 3. WebGPU initialisation

First, check for WebGPU support and request an adapter and device.

```js
// sim.js – Part 1: Setup
if (!navigator.gpu) {
  throw new Error('WebGPU not available');
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No adapter');
const device = await adapter.requestDevice();

const canvas = document.getElementById('sim-canvas');
const context = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'premultiplied' });
```

---

## 4. Particle data and buffers

We store positions and velocities interleaved in a single buffer (vec2<f32> for each). A second buffer holds the acceleration from the previous step – this is the “persistent acceleration” needed by velocity Verlet.

```js
const N = 4096;                     // number of particles
const WORKGROUP_SIZE = 64;
const DT_BASE = 0.0001;
const L = 2.0;                      // box half‑width
const DIAMETER = 0.01;
const STIFFNESS = 1000.0;
const V0 = 1.0;                     // initial speed

// Buffers: two copies for ping‑pong, one for accelerations
const particleByteSize = N * 4 * 4;   // pos (vec2) + vel (vec2) = 16 bytes/particle
const accByteSize = N * 2 * 4;        // acc (vec2) = 8 bytes/particle

const bufferA = device.createBuffer({
  size: particleByteSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const bufferB = device.createBuffer({ /* same */ });
const accBuffer = device.createBuffer({
  size: accByteSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
```

Initialize positions randomly inside the box (with a margin) and give each particle a fixed speed V0 in a random direction.

```js
function initialConditions() {
  const data = new Float32Array(N * 4);
  const margin = DIAMETER * 1.5;
  for (let i = 0; i < N; i++) {
    data[i*4+0] = (Math.random() - 0.5) * (L - 2*margin);
    data[i*4+1] = (Math.random() - 0.5) * (L - 2*margin);
    const theta = Math.random() * 2 * Math.PI;
    data[i*4+2] = V0 * Math.cos(theta);
    data[i*4+3] = V0 * Math.sin(theta);
  }
  return data;
}
device.queue.writeBuffer(bufferA, 0, initialConditions());
device.queue.writeBuffer(bufferB, 0, initialConditions());
device.queue.writeBuffer(accBuffer, 0, new Float32Array(N*2)); // zero acc
```

---

## 5. Velocity Verlet – two compute passes

Velocity Verlet splits the step into:

1. Kick‑drift: advance velocities to half‑step, then positions to full step.
2. Force + final kick: compute new accelerations from the new positions, then complete the velocity update.

We’ll write two WGSL compute shaders.

### Pass 1 – kick‑drift

```wgsl
struct Particle { pos: vec2<f32>, vel: vec2<f32> };
struct Params { dt: f32, diameter: f32, stiffness: f32, half_L: f32 };

@group(0) @binding(0) var<storage, read>       src: array<Particle>;
@group(0) @binding(1) var<storage, read_write> dst: array<Particle>;
@group(0) @binding(2) var<storage, read>       acc: array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&src)) { return; }

  let p = src[i];
  let a = acc[i];
  let dt = params.dt;

  var v_half = p.vel + 0.5 * a * dt;
  var x_new = p.pos + v_half * dt;

  // Reflecting walls
  let hL = params.half_L;
  if (x_new.x >  hL) { x_new.x =  2.0 * hL - x_new.x; v_half.x = -v_half.x; }
  if (x_new.x < -hL) { x_new.x = -2.0 * hL - x_new.x; v_half.x = -v_half.x; }
  if (x_new.y >  hL) { x_new.y =  2.0 * hL - x_new.y; v_half.y = -v_half.y; }
  if (x_new.y < -hL) { x_new.y = -2.0 * hL - x_new.y; v_half.y = -v_half.y; }

  dst[i].pos = x_new;
  dst[i].vel = v_half;
}
```

### Pass 2 – force + final kick

Forces are computed as an all‑pairs $O(N^2)$ loop. To keep memory access efficient, we load a tile of positions into workgroup memory.

```wgsl
@group(0) @binding(0) var<storage, read_write> state: array<Particle>;
@group(0) @binding(1) var<storage, read_write> acc:   array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             params: Params;

var<workgroup> tile: array<vec2<f32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&state);
  var my_pos = i < n ? state[i].pos : vec2(1e30);

  let d  = params.diameter;
  let d2 = d * d;
  var a_new = vec2(0.0);

  let tiles = (n + 63u) / 64u;
  for (var t = 0u; t < tiles; t++) {
    let j = t * 64u + lid.x;
    tile[lid.x] = j < n ? state[j].pos : vec2(1e30);
    workgroupBarrier();

    for (var k = 0u; k < 64u; k++) {
      let dr = my_pos - tile[k];
      let r2 = dot(dr, dr);
      if (r2 < d2 && r2 > 1e-12) {
        let r = sqrt(r2);
        let f = params.stiffness * (d - r) / r;
        a_new += dr * f;
      }
    }
    workgroupBarrier();
  }

  if (i < n) {
    let v_half = state[i].vel;
    state[i].vel = v_half + 0.5 * a_new * params.dt;
    acc[i] = a_new;
  }
}
```

JavaScript plumbing

Create the compute pipelines and bind groups. We’ll use ping‑pong buffering: after Pass 1 the “latest” state moves from buffer A to buffer B (or vice versa). Pass 2 then operates in‑place on the latest buffer.

```js
// … create shader modules, pipelines, and bind groups (see full code)
function recordVVStep(encoder) {
  // Pass 1: kick-drift src -> dst
  const pass1 = encoder.beginComputePass();
  pass1.setPipeline(pipeline1);
  pass1.setBindGroup(0, latestIsA ? p1BindAB : p1BindBA);
  pass1.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
  pass1.end();
  latestIsA = !latestIsA;

  // Pass 2: force + final kick on the latest buffer
  const pass2 = encoder.beginComputePass();
  pass2.setPipeline(pipeline2);
  pass2.setBindGroup(0, latestIsA ? p2BindA : p2BindB);
  pass2.dispatchWorkgroups(Math.ceil(N / WORKGROUP_SIZE));
  pass2.end();
}
```

---

## 6. Rendering – particles with a thermal palette

The render pipeline uses instanced drawing: a single quad (6 vertices) per particle. The vertex shader reads particle position and velocity, computes a colour from the speed via a 10‑stop palette, and passes it to the fragment shader.

Palette (in WGSL)

We embed the same palette used in the JavaScript code as a constant array.

```wgsl
fn palette(t: f32) -> vec3<f32> {
  let stops = array<vec3<f32>, 10>(
    vec3(0.91, 0.93, 0.96), // pale blue-white
    vec3(0.72, 0.77, 0.86),
    vec3(0.43, 0.53, 0.69),
    vec3(0.18, 0.27, 0.47), // navy
    vec3(0.21, 0.52, 0.60), // turquoise
    vec3(0.51, 0.65, 0.47),
    vec3(0.79, 0.65, 0.31),
    vec3(0.78, 0.49, 0.23),
    vec3(0.70, 0.22, 0.37),
    vec3(0.29, 0.10, 0.16)  // dark maroon
  );
  let tc  = clamp(t, 0.0, 1.0);
  let idx = tc * 9.0;
  let i0  = u32(floor(idx));
  let i1  = min(i0 + 1u, 9u);
  let a   = idx - f32(i0);
  return mix(stops[i0], stops[i1], a);
}
```

Vertex & fragment shaders

```wgsl
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0)       uv:  vec2<f32>,
  @location(1)       color: vec3<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32,
      @builtin(instance_index) ii: u32) -> VSOut {
  // Quad corners
  var corners = array<vec2<f32>, 6>(
    vec2(-1,-1), vec2(1,-1), vec2(-1,1),
    vec2(-1,1),  vec2(1,-1), vec2(1,1)
  );
  let p = particles[ii];
  let t = clamp(length(p.vel) / 2.8, 0.0, 1.0); // V_COLOR_MAX = 2.8

  var out: VSOut;
  out.pos   = vec4(p.pos + corners[vi] * 0.005, 0.0, 1.0); // radius = DIAMETER/2
  out.uv    = corners[vi];
  out.color = palette(t);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let r = length(in.uv);
  if (r > 1.0) { discard; }
  let alpha = smoothstep(1.0, 0.85, r);
  return vec4(in.color * alpha, alpha);
}
```

The render pipeline is straightforward: one bind group for the particle buffer, and a draw call with N instances.

```js
const renderPipeline = device.createRenderPipeline({ /* … */ });
const renderBindGroup = device.createBindGroup({ /* … */ });

function recordRender(encoder) {
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.031, g: 0.039, b: 0.078, a: 1.0 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
  });
  pass.setPipeline(renderPipeline);
  pass.setBindGroup(0, latestIsA ? renderBindA : renderBindB);
  pass.draw(6, N);
  pass.end();
}
```

---

## 7. Speed histogram and readback

To analyze the distribution, we copy the particle buffer to a mapped buffer once per frame, bin the speeds, and draw a histogram using the Canvas 2D API.

```js
const readbackBuffer = device.createBuffer({
  size: particleByteSize,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

function startReadback() {
  if (readbackInFlight) return;
  readbackInFlight = true;
  const src = latestIsA ? bufferA : bufferB;
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, readbackBuffer, 0, particleByteSize);
  device.queue.submit([encoder.finish()]);

  readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
    const data = new Float32Array(readbackBuffer.getMappedRange());
    // … bin speeds, update histCounts, compute kT/m
    readbackBuffer.unmap();
    readbackInFlight = false;
  });
}
```

The histogram is drawn with histCtx.fillRect() for each bin, and the theoretical Maxwell–Boltzmann curve

$$ f(v) = \frac{v}{k_B T/m} \exp\!\left(-\frac{v^2}{2\,k_B T/m}\right) $$

is overlaid in red. The measured kT/m is simply the mean kinetic energy per particle.

---

## 8. Animation loop and controls

The main loop runs requestAnimationFrame. Each frame we:

* Determine how many integration sub‑steps to perform (controlled by the “Speed” slider).
* Record one command encoder with the required number of recordVVStep calls, followed by recordRender.
* Submit the encoder and request the next frame.

```js
function loop() {
  // … update FPS display

  const encoder = device.createCommandEncoder();
  if (!paused) {
    for (let s = 0; s < substeps; s++) recordVVStep(encoder);
  }
  recordRender(encoder);
  device.queue.submit([encoder.finish()]);

  startReadback();
  drawHistogram();

  requestAnimationFrame(loop);
}
```

The Speed slider allows negative values – the signed dt makes the integrator run backward in time, demonstrating time‑reversibility.

```js
speedSlider.addEventListener('input', () => {
  const speed = parseFloat(speedSlider.value);
  substeps = Math.abs(Math.round(speed));
  dtSigned = (speed !== 0) ? Math.sign(speed) * DT_BASE : dtSigned;
  writeParams(); // update uniform buffer
  speedLabel.textContent = `${Math.round(speed)}×`;
});
```

---

## 9. Why velocity Verlet?

The original simulation used symplectic Euler, which is symplectic (energy oscillates rather than drifts) but only first‑order accurate. With the stiff soft‑sphere potential, the $\mathcal{O}(\Delta t)$ error in the modified Hamiltonian caused a visible drift in the temperature diagnostic.

Velocity Verlet is a Strang splitting of the same drift‑kick flows. It is second‑order accurate, time‑reversible, and symplectic. Its shadow Hamiltonian error is $\mathcal{O}(\Delta t^2)$, reducing the drift to a level dominated by the finite collision‑resolution time $\tau_c$. The cost is negligible on the GPU – one extra half‑kick and one persistent acceleration buffer.

The difference is immediately visible: with velocity Verlet the measured $k_B T/m$ stays within 1–2% of its initial value for the entire run.

---

## 10. Integrating with Hugo

To embed the simulator in a Hugo site, place the HTML and JavaScript in static/ (or use a shortcode). For example, create layouts/shortcodes/ideal-gas.html:

```html
<div class="ideal-gas-sim">
  <canvas id="sim-canvas" width="800" height="600"></canvas>
  <!-- controls etc. -->
</div>
<script type="module" src="{{ "js/ideal-gas.js" | relURL }}"></script>
```

Then in a Markdown post:

```markdown
{{</* ideal-gas */>}}
```

Make sure WebGPU is only requested on pages that actually use it.

---

## 11. Conclusion and further ideas

You now have a fully GPU‑accelerated molecular‑dynamics simulation running in the browser. This pattern – compute shaders for physics, render shaders for visualisation – is applicable to many other particle‑based simulations (flocking, SPH fluids, N‑body gravity).

Possible extensions:

* Add Barnes–Hut or a grid‑based neighbour list to scale to larger N.
* Replace the soft‑sphere with a Lennard‑Jones potential.
* Add a thermostat to study non‑equilibrium steady states.
* Visualize the velocity field with line integral convolution.
