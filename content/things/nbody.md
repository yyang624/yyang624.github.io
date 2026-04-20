---
title: "Ideal gas and the Maxwell-Boltzmann distribution"
date: 2026-04-18
simulation: "nbody"
math: true
summary: "4096 particles in a 2D box simulating a hard-sphere gas. A delta-function initial speed distribution thermalises to Maxwell-Boltzmann through binary elastic collisions on the GPU."
---

$N = 4096$ particles of equal mass $m$ in a square box $[-1, 1]^2$ with reflecting walls, interacting as **hard spheres**. Unlike soft-sphere potentials, these particles undergo instantaneous elastic collisions upon contact (distance $r < d$).

The simulation is integrated via a two-pass compute scheme per time step:

1. **Drift & Boundary Pass**: Particles move linearly according to their velocity $\mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}_n \Delta t$. Positions are mirrored and normal velocities flipped upon contact with walls.

2. **Collision Pass**: A direct $\mathcal{O}(N^2)$ scan identifies overlapping pairs. To ensure exact conservation of momentum and kinetic energy on the GPU, the system uses a **mutual-best partner** logic: each particle identifies its most "urgent" approaching neighbour; an impulse is fired only if the preference is mutual. The velocity update follows the elastic impulse rule:

$$
\mathbf{v}_i \leftarrow \mathbf{v}_i - \frac{(\mathbf{v}_i - \mathbf{v}_j) \cdot (\mathbf{x}_i - \mathbf{x}_j)}{\|\mathbf{x}_i - \mathbf{x}_j\|^2} (\mathbf{x}_i - \mathbf{x}_j)
$$

This discrete-time approach is approximately time-reversible, though subject to finite-$\Delta t$ errors in collision timing.

Initial conditions place every particle at a fixed speed $v_0$ with a random direction, resulting in an initial speed PDF of $f_0(v) = \delta(v - v_0)$. Through binary collisions, the system redistributes kinetic energy and relaxes toward the two-dimensional Maxwell-Boltzmann distribution:

$$
f_\mathrm{MB}(v) = \frac{m v}{k_B T} \exp\!\left(-\frac{m v^2}{2 k_B T}\right),
$$

where the temperature $T$ is determined by the conserved initial kinetic energy per particle, $\frac{1}{2} m \langle v^2 \rangle = k_B T$.

### Visual Mapping
Particle colour maps speed $v$ linearly onto a ten-stop thermal palette, where $v=0$ is the coldest stop and $v \ge 2.6$ is saturated to the hottest stop. As the system thermalises, the initial monochromatic state evolves into the full spectrum defined below:

| Stop | Description | Normalized RGB | 8-bit RGB | Hex | Swatch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Coldest (Pale blue-white) | `[0.91, 0.93, 0.96]` | `(232, 237, 245)` | `#E8EDF5` | <span style="display:block; width:1.2em; height:1.2em; background-color:#E8EDF5; border:1px solid #ccc; margin:0 auto;"></span> |
| 1 | Light gray-blue | `[0.72, 0.77, 0.86]` | `(184, 196, 219)` | `#B8C4DB` | <span style="display:block; width:1.2em; height:1.2em; background-color:#B8C4DB; border:1px solid #ccc; margin:0 auto;"></span> |
| 2 | Medium blue | `[0.43, 0.53, 0.69]` | `(110, 135, 176)` | `#6E87B0` | <span style="display:block; width:1.2em; height:1.2em; background-color:#6E87B0; border:1px solid #ccc; margin:0 auto;"></span> |
| 3 | Navy (Freezing divide) | `[0.18, 0.27, 0.47]` | `(46, 69, 120)` | `#2E4578` | <span style="display:block; width:1.2em; height:1.2em; background-color:#2E4578; border:1px solid #ccc; margin:0 auto;"></span> |
| 4 | Turquoise | `[0.21, 0.52, 0.60]` | `(54, 133, 153)` | `#368599` | <span style="display:block; width:1.2em; height:1.2em; background-color:#368599; border:1px solid #ccc; margin:0 auto;"></span> |
| 5 | Lime-teal | `[0.51, 0.65, 0.47]` | `(130, 166, 120)` | `#82A678` | <span style="display:block; width:1.2em; height:1.2em; background-color:#82A678; border:1px solid #ccc; margin:0 auto;"></span> |
| 6 | Mellow yellow | `[0.79, 0.65, 0.31]` | `(201, 166, 79)` | `#C9A64F` | <span style="display:block; width:1.2em; height:1.2em; background-color:#C9A64F; border:1px solid #ccc; margin:0 auto;"></span> |
| 7 | Warm orange/gold | `[0.78, 0.49, 0.23]` | `(199, 125, 59)` | `#C77D3B` | <span style="display:block; width:1.2em; height:1.2em; background-color:#C77D3B; border:1px solid #ccc; margin:0 auto;"></span> |
| 8 | Scorching pink/red | `[0.70, 0.22, 0.37]` | `(179, 56, 94)` | `#B3385E` | <span style="display:block; width:1.2em; height:1.2em; background-color:#B3385E; border:1px solid #ccc; margin:0 auto;"></span> |
| 9 | Very hot dark maroon | `[0.29, 0.10, 0.16]` | `(74, 26, 41)` | `#4A1A29` | <span style="display:block; width:1.2em; height:1.2em; background-color:#4A1A29; border:1px solid #ccc; margin:0 auto;"></span> |

The panel below shows the running speed histogram (64 bins) against the theoretical $f_\mathrm{MB}$ curve. The histogram is updated by copying the velocity buffer back to the CPU via an asynchronous `mapAsync` readback. An exponential moving average ($\alpha = 0.4$) is applied to smooth the per-frame Poisson noise.
