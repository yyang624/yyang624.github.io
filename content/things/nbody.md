---
title: "Ideal gas and the Maxwell-Boltzmann distribution"
date: 2025-01-15
simulation: "nbody"
math: true
summary: "4096 particles in a 2D box. Starting from a delta-function speed distribution, elastic soft-sphere collisions drive the system to Maxwell-Boltzmann equilibrium."
---

$N = 4096$ particles of equal mass $m$ in the square box $[-1, 1]^2$
with reflecting walls, interacting through a short-range repulsive
soft-sphere pair potential

$$
U(r) = \tfrac{1}{2} k (d - r)^2 \quad (r < d), \qquad U(r) = 0 \quad (r \geq d),
$$

where $d$ is the particle diameter and $k$ the stiffness. The pair
force is evaluated on the GPU by direct $\mathcal{O}(N^2)$ summation
and integrated with symplectic Euler; dynamics are energy-conserving
to integrator order. Soft-sphere rather than true hard-sphere
collisions are used because hard spheres require event-driven
scheduling that does not parallelise efficiently on compute shaders.

All particles start at the same speed $v_0$ with random orientation,
so the initial speed distribution is $f_0(v) = \delta(v - v_0)$.
Elastic collisions redistribute kinetic energy between particles, and
by the Boltzmann $H$-theorem the system relaxes to the two-dimensional
Maxwell-Boltzmann distribution

$$
f_\mathrm{MB}(v) = \frac{m v}{k_B T} \exp\!\left(-\frac{m v^2}{2 k_B T}\right),
$$

whose temperature is fixed by equipartition,
$\tfrac{1}{2} m \langle v^2 \rangle = k_B T$, and therefore by the
(conserved) initial kinetic energy per particle.

The panel below shows the running speed histogram against
$f_\mathrm{MB}$ computed from the instantaneous $\langle v^2 \rangle$.
Relaxation proceeds over $\mathcal{O}(10)$ collision times per
particle; press **Reset** to rerun from the delta initial condition.
