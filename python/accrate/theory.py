"""Exact accumulation metrics for any design in acc_designs, from its low-part tables.

For a design with low-part width k (q = 2^k), the next accumulator value from
low state x and operand residue u (high parts zero) gives the next state
z(x,u) and carry c(x,u).  Under iid uniform residues the state is a Markov chain
and the accumulated error S_n = sum(q c_i - u_i) + X_n - X_0 (carry conservation).

Reported per design:
  drift      lim E[S_n]/n
  diffusion  lim Var(S_n)/n            (asymptotic variance)
  S          recurrent states reached from reset
  bound      ((q/S)^2 - 1)/12          (state-count lower bound on diffusion)
  mse_n      E[(S_n - b_n)^2] with the best constant compensation b_n = round(E S_n)
"""
import numpy as np

from acc_designs import evo_support, step_fn


def low_tables(design):
    kind, _, arg = design.partition("_")
    if kind == "exact":
        k = 0
    elif kind == "evo":
        k = evo_support(arg)
    else:
        k = int(arg)
    q = 1 << k
    f = step_fn(design)
    x, u = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    nxt = f(x.ravel(), u.ravel()).reshape(q, q)
    return k, nxt % q, nxt >> k


def _kernels(z, c, pmf=None):
    q = z.shape[0]
    x, u = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    w = np.full(q, 1.0 / q) if pmf is None else np.asarray(pmf, float) / np.sum(pmf)
    wu = w[u]
    e = (q * c + z - x - u).astype(float)
    P = np.zeros((q, q)); A = np.zeros((q, q)); B = np.zeros((q, q))
    np.add.at(P, (x, z), wu)
    np.add.at(A, (x, z), e * wu)
    np.add.at(B, (x, z), e ** 2 * wu)
    return P, A, B


def moments(z, c, ns, pmf=None, x0=0):
    """Exact (E S_n, Var S_n) from low state x0 for each n in ns; iid residues with law pmf (default uniform)."""
    P, A, B = _kernels(z, c, pmf)
    q = z.shape[0]
    v = np.zeros(q); v[x0 % q] = 1; m = np.zeros(q); h = np.zeros(q)
    out, want = {}, set(ns)
    for i in range(1, max(ns) + 1):
        v, m, h = v @ P, m @ P + v @ A, h @ P + 2 * m @ A + v @ B
        if i in want:
            out[i] = (m.sum(), h.sum() - m.sum() ** 2)
    return out


def chain_analysis(z, c, pmf=None, x0=0):
    """Exact long-run analysis on the recurrent class reached from x0.

    Returns the number of recurrent classes reachable from x0, the class size S,
    the maximal out-degree, drift, diffusion via the Poisson equation, the
    entropy rate h = H(X_i | X_{i-1}), and the entropy bound (q^2 2^{-2h}-1)/12
    (valid for uniform residues).
    """
    q = z.shape[0]
    w = np.full(q, 1.0 / q) if pmf is None else np.asarray(pmf, float) / np.sum(pmf)
    supp = np.nonzero(w > 0)[0]
    succ = [set(z[x, supp].tolist()) for x in range(q)]

    def reach(s):
        seen, st = {s}, [s]
        while st:
            for y in succ[st.pop()]:
                if y not in seen:
                    seen.add(y); st.append(y)
        return seen
    R0 = reach(x0)
    closed = [x for x in R0 if all(x in reach(y) for y in reach(x))]
    classes = {frozenset(reach(x)) for x in closed}
    cls = sorted(next(iter(classes)))
    idx = {x: i for i, x in enumerate(cls)}
    n = len(cls)
    P = np.zeros((n, n)); Y = np.zeros((n, q))
    for x in cls:
        for u in supp:
            P[idx[x], idx[z[x, u]]] += w[u]
        Y[idx[x]] = q * c[x] - np.arange(q)
    ev, vec = np.linalg.eig(P.T)
    pi = np.real(vec[:, np.argmin(abs(ev - 1))]); pi /= pi.sum()
    g = Y @ w
    mu = float(pi @ g)
    phi = np.linalg.lstsq(np.vstack([np.eye(n) - P, pi]), np.concatenate([g - mu, [0.0]]), rcond=None)[0]
    var = h = 0.0
    for x in cls:
        V = np.array([Y[idx[x], u] + phi[idx[z[x, u]]] for u in range(q)])
        var += pi[idx[x]] * float(w @ (V - w @ V) ** 2)
        m = np.zeros(n)
        np.add.at(m, [idx[z[x, u]] for u in supp], w[supp])
        m = m[m > 0]
        h += pi[idx[x]] * float(-(m * np.log2(m)).sum())
    return {"classes": len(classes), "S": n, "outdeg": max(len(succ[x]) for x in cls), "drift": mu,
            "diffusion": var, "entropy_rate": h, "entropy_bound": (q * q * 2 ** (-2 * h) - 1) / 12}


def recurrent_count(z):
    q = z.shape[0]
    succ = [set(z[x].tolist()) for x in range(q)]

    def reach(s):
        seen, st = {s}, [s]
        while st:
            for y in succ[st.pop()]:
                if y not in seen:
                    seen.add(y); st.append(y)
        return seen
    from_reset = reach(0)
    return sum(1 for x in from_reset if all(x in reach(y) for y in reach(x)))


def metrics(design, ns=(16, 27, 64, 144, 256, 576, 1024, 4096)):
    k, z, c = low_tables(design)
    q = 1 << k
    lo, hi = 4096, 8192
    mo = moments(z, c, sorted(set(ns) | {lo, hi}))
    ca = chain_analysis(z, c)
    diffusion = ca["diffusion"]
    # cross-check the Poisson solution against the finite-difference estimate
    fd = (mo[hi][1] - mo[lo][1]) / (hi - lo)
    assert abs(fd - diffusion) <= 1e-6 * max(1.0, diffusion), (design, fd, diffusion)
    assert ca["classes"] == 1, (design, "several recurrent classes reachable from reset")
    row = {"design": design, "k": k, "drift": ca["drift"], "diffusion": diffusion, "S": ca["S"],
           "outdeg": ca["outdeg"], "entropy_rate": ca["entropy_rate"], "entropy_bound": ca["entropy_bound"],
           "bound": ((q / ca["S"]) ** 2 - 1) / 12, "state_blind": bool(np.all(c == c[0:1, :])),
           "t_eq": float(np.log2(12 * diffusion + 1) / 2)}
    for n in ns:
        mean, var = mo[n]
        row[f"mse_raw_{n}"] = var + mean ** 2
        row[f"mse_comp_{n}"] = var + (mean - round(mean)) ** 2
    return row
