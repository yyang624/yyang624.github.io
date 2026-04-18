---
title: "A note on the symplectic Euler scheme"
date: 2025-04-15
math: true
---

Writing the N-body simulation on the Things page reminded me why
symplectic Euler is nice even though it's only first-order accurate.

For a Hamiltonian $H(q, p) = T(p) + V(q)$, the update

$$
p_{n+1} = p_n - \Delta t\, \nabla V(q_n), \qquad
q_{n+1} = q_n + \Delta t\, \nabla T(p_{n+1}),
$$

is the composition of two exact flows. As such, it is a symplectic map and
conserves a *modified* Hamiltonian $\tilde{H} = H + \Delta t \cdot H_1 + \dots$
to all orders in $\Delta t$. The true energy $H$ oscillates around its
initial value instead of drifting — the standout property compared to
explicit Euler, which leaks energy monotonically.

The cost: only first-order accuracy in $\Delta t$. For production work,
go to velocity-Verlet (second-order, still symplectic, same cost per step).
