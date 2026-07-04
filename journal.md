# Experiments Journal

Date : July 4, 2026
--
- $\beta$ changed from `0.1` to `1.0`
- While optimizing $\mu$ for $\min_\mu h_\mu(s, a, x)-d_{target}(s, a, x, b)$, $d_{target}=|r-y|+\gamma g(s', x')$ where $u(s, x)=\max(g(s, x), g(x, s))$
- In contrast to the $U^\pi \approx \frac{1}{2}(\|\phi(x)\|^2+\|\phi(s)\|^2) -\lambda \theta(\phi(s), \phi(x))$, we empirically noted that our algorithm imparts non-zero self distance. So I am hoping it's a good sign
- Experimented with 3 seeds, and our algorithm outperforms SAC in `HumanoidStand`

Things need to be worked upon :
- Need to reproduce the results of SAC in `CheetahRun`. Subsequently need to test our algorithm on top of this 
- In `HumanoidRun` at $\text{step}\approx 36$, the $Loss_U$ i.e the state metric loss becomes exactly $0.0$ , even though it outperforms the vanilla sac, but what's going on ? To be noted : only tested for `seed=0`

Everlong Question

- A model free way of learning the homomorphic metric guarantees convergence only asymptotically. Incidentally, the policy also reaches the optimality at the same rate. 
    - why would someone introduce this `metric-learning` complexity ? 
    - what is the trade-off between the `return vs complexity` of introducing the algorithm ?
- I think homomorphism has to offer where one is looking to `transfer`. 
    - Exploiting `symmetry` in the same MDP is like transfering and mapping skills to equivalent states
    - We should be able to do that between MDPs as well. Don't you think so ?