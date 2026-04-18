---
title: "Ideal gas and the Maxwell-Boltzmann distribution"
date: 2026-04-18
simulation: "nbody"
math: true
summary: "4096 particles in a 2D box, integrated with velocity Verlet. A delta-function initial speed distribution thermalises to Maxwell-Boltzmann through elastic soft-sphere collisions."
---

$N = 4096$ particles of equal mass $m$ in the square box $[-1, 1]^2$
with reflecting walls, interacting through a short-range repulsive
soft-sphere pair potential

$$
U(r) = \tfrac{1}{2} k (d - r)^2 \quad (r < d), \qquad U(r) = 0 \quad (r \geq d),
$$

where $d$ is the particle diameter and $k$ the stiffness. The pair
force is evaluated on the GPU by direct $\mathcal{O}(N^2)$ summation and
integrated with **velocity Verlet**:

$$
\begin{aligned}
\mathbf{v}_{n+1/2} &= \mathbf{v}_n + \tfrac{\Delta t}{2}\, \mathbf{a}_n, \\
\mathbf{x}_{n+1}   &= \mathbf{x}_n + \Delta t\, \mathbf{v}_{n+1/2}, \\
\mathbf{a}_{n+1}   &= \mathbf{F}(\mathbf{x}_{n+1}) / m, \\
\mathbf{v}_{n+1}   &= \mathbf{v}_{n+1/2} + \tfrac{\Delta t}{2}\, \mathbf{a}_{n+1}.
\end{aligned}
$$

This is second-order accurate, time-reversible, and symplectic; total
energy fluctuates at $\mathcal{O}(\Delta t^2)$ with no secular drift.
The scheme is realised as two compute passes per step (kick-drift, then
force-evaluation and final kick), with a persistent acceleration buffer
holding $\mathbf{a}_n$ between steps.

Initial conditions place every particle at speed $v_0$ with a random
direction, so the initial speed PDF is $f_0(v) = \delta(v - v_0)$.
Elastic collisions redistribute kinetic energy between particles, and
by the Boltzmann $H$-theorem the system relaxes to the two-dimensional
Maxwell-Boltzmann distribution

$$
f_\mathrm{MB}(v) = \frac{m v}{k_B T} \exp\!\left(-\frac{m v^2}{2 k_B T}\right),
$$

whose temperature is fixed by equipartition,
$\tfrac{1}{2} m \langle v^2 \rangle = k_B T$, and therefore by the
(conserved) initial kinetic energy per particle.

Particle colour maps speed onto a ten-stop thermal palette (pale
blue-white at $v = 0$, navy at a "dividing wall" just below the
most-probable speed, turquoise through yellows and oranges to deep red
at the tail of the distribution). The initial configuration is
monochromatic; as the system thermalises it acquires the full colour
range.

The panel below shows the running speed histogram (64 bins, bars
coloured by the same palette) against $f_\mathrm{MB}$ computed from the
instantaneous $\langle v^2 \rangle$. The histogram is sampled by copying
the velocity buffer back to CPU memory on every frame through an
asynchronous `mapAsync` readback; the in-flight guard throttles the
effective update rate to the GPU round-trip ($\sim 25$–$30$ Hz), and a
light exponential moving average ($\alpha = 0.4$) smooths the per-sample
Poisson noise.
