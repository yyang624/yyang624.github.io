---
title: "Derivation of the Boltzmann factor"
date: 2026-04-20
tags: ["statistical mechanics", "thermodynamics"]
math: true
---
This note derives the Boltzmann factor
$$P(\epsilon_i)\propto\exp\left(-\frac{\epsilon_i}{k_B T}\right),$$
the probability that a single particle in an isolated system at thermal equilibrium occupies a state of energy $\epsilon_i$.

**Fundamental assumptions:**
  1. The system is isolated, with fixed total energy $E_t$ and fixed particle number $N$.
  2. Every microstate of the isolated system is equally likely.

The probability that a specific particle is in a state of energy $\epsilon_i$ equals the number of microstates available to the rest of the system divided by the total number of microstates:
$$P(\epsilon_i)=\frac{\Omega(N-1,E_t-\epsilon_i)}{\Omega(N,E_t)},$$
where $\Omega(N,E)$ denotes the multiplicity at fixed $N$ and $E$.

In the thermodynamic limit, $\Omega(N,E)$ is astronomically large, so it is convenient to work with its logarithm:
$$\ln P(\epsilon_i)= \ln\Omega(N-1,E_t-\epsilon_i) - \ln\Omega(N,E_t).$$
Since $\epsilon_i \ll E_t$, we Taylor-expand about $E_t$:
$$\ln\Omega(N-1,E_t-\epsilon_i)\approx \ln\Omega(N-1,E_t)-\left.\frac{\partial \ln\Omega}{\partial E}\right|_{E_t}\epsilon_i,$$
which gives
$$\ln P(\epsilon_i)\approx\ln\Omega(N-1,E_t)-\ln\Omega(N,E_t)-\frac{\partial\ln\Omega}{\partial E}\epsilon_i.$$
The first two terms are independent of $\epsilon_i$ and can be absorbed into a constant $\ln C$:
$$P(\epsilon_i)\approx C\exp\left(-\frac{\partial\ln\Omega}{\partial E}\epsilon_i\right).$$
At thermal equilibrium,
$$\frac{\partial \ln\Omega}{\partial E} \equiv \frac{1}{k_B T}$$
defines the temperature $T$, with $k_B$ the Boltzmann constant. We thereby recover the Boltzmann factor
$$P(\epsilon_i)\propto\exp\left(-\frac{\epsilon_i}{k_B T}\right).$$

{{< theorem name="Statistical definition of temperature" >}}
Consider an isolated system with fixed total energy $E_t$ partitioned into subsystems $A$ and $B$. The total multiplicity factorizes:
$$\Omega_{t}(E_{t})=\Omega_A(E_A)\Omega_B(E_B),\qquad E_B=E_t-E_A.$$
By the equal-a-priori-probability postulate, the system is most likely found in the configuration that maximizes $\Omega_t$, so
$$\frac{d\Omega_t}{dE_A}=\frac{d\Omega_A(E_A)}{dE_A}\Omega_B(E_B)+\Omega_A(E_A)\frac{d\Omega_B(E_B)}{dE_A}=0.$$
Using $dE_B/dE_A=d(E_t-E_A)/dE_A=-1$,
$$\frac{d\Omega_B(E_B)}{dE_A}=\frac{d\Omega_B(E_B)}{dE_B}\frac{dE_B}{dE_A}=-\frac{d\Omega_B(E_B)}{dE_B},$$
so the extremum condition becomes
$$\frac{1}{\Omega_A}\frac{d\Omega_A}{dE_A}=\frac{1}{\Omega_B}\frac{d\Omega_B}{dE_B}.$$
This holds for any partitioning, so
$$\frac{1}{\Omega}\frac{d\Omega}{dE}=\frac{d\ln\Omega}{dE}\equiv\frac{1}{k_B T}$$
defines the temperature $T$, an intensive quantity characterizing the equilibrium state.
{{< /theorem >}}
