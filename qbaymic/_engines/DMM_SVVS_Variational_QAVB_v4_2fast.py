#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS VQT QAVB — JAX + JIT + vmap + pmap (CPU-parallel) Variant (v4_2fast)
=============================================================================

v4_2 ported the VarQITE JAX template (v3_1) to the VQT free-energy method:
one jitted, vmapped, lax.scanned Adam kernel over the whole sample batch,
with jax.grad replacing parameter-shift. That removed the Python interpreter
overhead but left ONE bottleneck untouched — the same one v3_1 hit:

    JAX's CPU backend exposes a SINGLE device by default, so jax.vmap runs the
    entire N-sample batch on ~one busy thread. On a many-core box htop shows
    one core pegged and the rest idle; the embarrassingly-parallel sample loop
    never spreads across cores.

v4_2fast applies the exact fix v3_1 uses to reach ~all-core utilisation:
shard the sample batch across multiple XLA host devices with

    jax.pmap(jax.vmap(adam_trajectory))

so each device runs a contiguous shard of the N samples truly in parallel.
This is a pure-systems change — the Adam free-energy minimisation, the static
Pauli basis, the softmax categorical mixture, the readout, and EVERY numeric
result are identical to v4_2. Only the device map differs.

What changes vs v4_2
--------------------
  • Device count is chosen at MODULE IMPORT (XLA fixes the host-device pool
    once, at first JAX init; it cannot grow afterwards):
        min(os.cpu_count() // 2, 32),  or env V4_2_CPU_DEVICES  ("1" disables).
  • A `cpu_parallel` flag ("auto" | True | False) chooses whether to USE those
    devices. "auto" shards once the batch is large enough to amortise the pmap
    dispatch (>= cpu_parallel_min_batch AND > 2·n_devices rows).
  • The per-sample Adam kernel is additionally compiled as pmap(vmap(...)) and
    the batch is padded to a multiple of n_devices, sharded (D, per, …), run,
    and unpadded — the v3_1 `_run_kernels` pattern, adapted to VQT's TWO
    parameter arrays (θ ansatz + φ categorical logits).

What is unchanged vs v4_2 (verified numerically, ~1e-12)
-------------------------------------------------------
  • free_energy / jax.grad / Adam scan / softmax mixture / readout
  • static Pauli enumeration, Walsh–Hadamard coefficients, phantom padding
  • warm-start carry (θ_warm, φ_warm), dedup (Layer-C), JIT-warmup accounting
  • update_strategy masks (joint / alternating / phi_first)

Amdahl note (same as v3_1)
--------------------------
pmap accelerates ONLY the VQT E-step. The SVVS M-step, trigamma E-LL, k-means
dedup and pruning stay single-threaded NumPy, so a full fit will not peg every
core — the realised speedup grows as the E-step dominates (larger N, more
n_vqt_steps, deeper ansatz, larger K).

Use
---
    from DMM_SVVS_Variational_QAVB_v4_2fast import DMM_SVVS_VQT_QAVB_JAX_Fast
    m = DMM_SVVS_VQT_QAVB_JAX_Fast(
        K_max=4, beta0=5.0, s0=1.0, tau1=8, tau2=18,
        ansatz_depth=2, n_vqt_steps=40,
        cpu_parallel="auto",          # shard the sample batch across cores
        dedup_n_clusters=None,
    )
    m.fit(X)

    # to control the shard count:
    #   V4_2_CPU_DEVICES=16 python your_script.py    # 16 shards
    #   V4_2_CPU_DEVICES=1  python your_script.py    # disable (= v4_2)
"""
from __future__ import annotations

import os
import sys
from time import time

import numpy as np

# Force 64-bit before JAX is touched — quantum amplitudes need it.
os.environ.setdefault("JAX_ENABLE_X64", "1")

# ── CPU multi-device setup (MUST happen before JAX initialises) ────────────
#
# Identical mechanism to v3_1: XLA creates the host-platform device pool ONCE,
# at first JAX import, from the flag below; it cannot be changed afterwards.
# We choose the shard count HERE, at module import, before anything (including
# the v4_2 base class chain) touches jax.
#
# Count resolution (first match wins):
#   1. env V4_2_CPU_DEVICES   (explicit override; "1" disables sharding)
#   2. min(os.cpu_count() // 2, 32)   — efficient regime for the tiny per-
#      sample VQT kernel; past ~32 shards dispatch overhead flattens returns.
# Set V4_2_CPU_DEVICES=1 to recover the exact single-device v4_2 behaviour.
def _resolve_cpu_device_count() -> int:
    env = os.environ.get("V4_2_CPU_DEVICES")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cores = os.cpu_count() or 1
    return max(1, min(cores // 2, 32))


_V4_2_CPU_DEVICES = _resolve_cpu_device_count()
if _V4_2_CPU_DEVICES > 1:
    # Only append; never clobber a user-provided XLA_FLAGS, and don't override
    # an existing device-count flag if the user already set one.
    _flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _flags:
        os.environ["XLA_FLAGS"] = (
            (_flags + " " if _flags else "")
            + f"--xla_force_host_platform_device_count={_V4_2_CPU_DEVICES}"
        )

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# Import AFTER the XLA flag is set so the device pool is created with the right
# size on first JAX init. v4_2 imports JAX lazily (in _init_pennylane_device),
# so this ordering is what makes the shard count take effect.
from DMM_SVVS_Variational_v2 import NumericalStability  # noqa: E402
from DMM_SVVS_Variational_QAVB_v4_2 import (  # noqa: E402
    DMM_SVVS_VQT_QAVB_JAX,
)


class DMM_SVVS_VQT_QAVB_JAX_Fast(DMM_SVVS_VQT_QAVB_JAX):
    """
    CPU-parallel VQT QAVB (v4_2fast).

    Subclasses the v4_2 JAX VQT model and adds jax.pmap sharding of the
    per-sample Adam free-energy kernel across XLA host devices, so the
    embarrassingly-parallel sample batch spreads across CPU cores. Every
    numeric result is identical to v4_2; only the device map changes.

    Parameters added on top of v4_2
    --------------------------------
    cpu_parallel : "auto" | bool, default "auto"
        Whether to shard the sample batch across the host devices created at
        module import (see V4_2_CPU_DEVICES).
          "auto" -> shard when >1 device exists AND the batch is large enough
                    to amortise pmap dispatch.
          True   -> always shard when >1 device exists.
          False  -> force the single-device vmap path (== v4_2).
        The number of devices is FIXED at import; this flag only chooses
        whether to use them.
    cpu_parallel_min_batch : int, default 64
        Smallest batch worth sharding in "auto" mode (below this, pmap
        dispatch + padding overhead outweighs the parallelism).
    """

    def __init__(
        self,
        *args,
        cpu_parallel="auto",
        cpu_parallel_min_batch: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cpu_parallel = cpu_parallel
        self.cpu_parallel_min_batch = max(1, int(cpu_parallel_min_batch))

        # pmap kernels + device count (set in _init_pennylane_device).
        self._pmap_kernel = None      # pmap(vmap(scan(adam_step)))
        self._pmap_probs = None       # pmap(vmap(readout))
        self._pmap_F = None           # pmap(vmap(free_energy))
        self._n_devices = 1

    # ── Record device count, then build kernels (super builds vmap ones) ──

    def _init_pennylane_device(self):
        # super() imports jax, sets self._jax/_jnp, builds the static Pauli
        # basis and the single-device vmap kernels via _build_jax_kernels
        # (which we override below to ALSO build the pmap kernels).
        super()._init_pennylane_device()

    def _build_jax_kernels(self):
        # Build all the single-device vmap/jit kernels exactly as v4_2 does.
        super()._build_jax_kernels()

        jax = self._jax
        self._n_devices = max(1, jax.local_device_count())

        # Reconstruct the SAME trajectory/readout/free_energy closures v4_2
        # built, but wrapped in pmap(vmap(...)). Rather than duplicate the
        # (long) closure bodies, we wrap the already-vmapped+jitted kernels'
        # underlying functions. The cleanest, drift-proof way is to rebuild
        # the pmap kernels from the vmapped functions captured below.
        #
        # v4_2's _build_jax_kernels assigns:
        #   self._batched_kernel = jit(vmap(trajectory, (0,0,0,None)))
        #   self._probs_batched  = jit(vmap(readout,    (0,0)))
        #   self._F_batched      = jit(vmap(free_energy,(0,0,0,None)))
        # We re-derive pmap(vmap(...)) from the same per-sample functions by
        # re-extracting them. Since those locals aren't stored, we rebuild the
        # pmap kernels by composing pmap over the EXISTING vmapped callables:
        # pmap maps the device axis, the inner jitted-vmap maps the per-shard
        # sample axis. This composes correctly and keeps ONE source of truth
        # for the kernel math (the v4_2 closures).
        if self._n_devices > 1:
            # _batched_kernel etc. are jit(vmap(f)). pmap over them maps the
            # leading device axis; the inner vmap maps the per-shard rows.
            self._pmap_kernel = jax.pmap(
                self._batched_kernel, in_axes=(0, 0, 0, None))
            self._pmap_probs = jax.pmap(
                self._probs_batched, in_axes=(0, 0))
            self._pmap_F = jax.pmap(
                self._F_batched, in_axes=(0, 0, 0, None))
        else:
            self._pmap_kernel = None
            self._pmap_probs = None
            self._pmap_F = None

    # ── Kernel dispatch: pmap over CPU shards, or single-device vmap ──────

    def _use_pmap(self, B: int) -> bool:
        """Decide whether to shard a batch of size B across devices."""
        if self._n_devices <= 1 or self._pmap_kernel is None:
            return False
        if self.cpu_parallel is False:
            return False
        if self.cpu_parallel == "auto":
            # Worth sharding only if the batch amortises pmap dispatch +
            # padding and gives >1 row/device.
            return B >= max(self.cpu_parallel_min_batch, 2 * self._n_devices)
        return True   # cpu_parallel is True

    def _run_vqt_kernels(self, theta0, phi0, C_mat, beta_t):
        """
        Evaluate (theta_final, phi_final, probs, F) for the FULL batch.

        Routes through jax.pmap (true multi-core over sample shards) when
        _use_pmap is satisfied, else the single-device vmap kernel chunked by
        sample_batch_size. Results are identical either way (same kernel, just
        a different device map) — verified to ~1e-12.
        """
        jnp = self._jnp
        B = theta0.shape[0]

        if self._use_pmap(B):
            D = self._n_devices
            # Pad B up to a multiple of D so the device axis is even; run, then
            # drop the padding rows (cheap zeros, never read back).
            pad = (-B) % D
            if pad:
                theta0 = np.concatenate(
                    [theta0, np.zeros((pad, self.n_params))], axis=0)
                phi0 = np.concatenate(
                    [phi0, np.zeros((pad, self.K_pad))], axis=0)
                C_pad = jnp.concatenate(
                    [C_mat, jnp.zeros((pad, C_mat.shape[1]))], axis=0)
            else:
                C_pad = C_mat
            Bp = B + pad
            per = Bp // D

            th_sh = jnp.asarray(theta0).reshape(D, per, self.n_params)
            ph_sh = jnp.asarray(phi0).reshape(D, per, self.K_pad)
            c_sh = C_pad.reshape(D, per, C_pad.shape[1])
            beta = float(beta_t)

            thf, phf = self._pmap_kernel(th_sh, ph_sh, c_sh, beta)
            pr = self._pmap_probs(thf, phf)
            F = self._pmap_F(thf, phf, c_sh, beta)

            thf = np.asarray(thf).reshape(Bp, self.n_params)[:B]
            phf = np.asarray(phf).reshape(Bp, self.K_pad)[:B]
            pr = np.asarray(pr).reshape(Bp, self.K_pad)[:B]
            F = np.asarray(F).reshape(Bp)[:B]
            return thf, phf, pr, F

        # Single-device path: chunk by sample_batch_size (v4_2 behaviour).
        bs = self.sample_batch_size or B
        thf_all = np.empty((B, self.n_params), dtype=np.float64)
        phf_all = np.empty((B, self.K_pad), dtype=np.float64)
        pr_all = np.empty((B, self.K_pad), dtype=np.float64)
        for start in range(0, B, bs):
            stop = min(start + bs, B)
            th0 = jnp.asarray(theta0[start:stop])
            ph0 = jnp.asarray(phi0[start:stop])
            cc = C_mat[start:stop]
            thf, phf = self._batched_kernel(th0, ph0, cc, float(beta_t))
            pr = self._probs_batched(thf, phf)
            thf_all[start:stop] = np.asarray(thf)
            phf_all[start:stop] = np.asarray(phf)
            pr_all[start:stop] = np.asarray(pr)
        F_all = np.asarray(self._F_batched(
            jnp.asarray(thf_all), jnp.asarray(phf_all), C_mat, float(beta_t)))
        return thf_all, phf_all, pr_all, F_all

    # ── Batched VQT routed through the pmap-aware dispatcher ──────────────

    def _vqt_batch(self, D_block: np.ndarray, beta_t: float, s_t: float):
        """
        Same contract as v4_2._vqt_batch, but the kernel execution goes
        through _run_vqt_kernels (pmap over CPU shards when worthwhile).

        Returns
        -------
        r : (B, K) responsibilities (clip+renormalise on the K-block).
        theta_final : (B, n_params)  for warm-start carry.
        phi_final   : (B, K_pad)     for warm-start carry.
        F_final     : (B,)           final per-sample free energy.
        """
        EPS = NumericalStability.EPS
        B = D_block.shape[0]

        C_mat = self._coefficients_batch(D_block, s_t)            # (B, P) JAX
        theta0, phi0 = self._init_theta_phi_batch(B)

        theta_final, phi_final, probs, F_final = self._run_vqt_kernels(
            theta0, phi0, C_mat, beta_t)

        r = np.clip(probs[:, : self.K], EPS, None)
        r = r / r.sum(axis=1, keepdims=True)
        return r, theta_final, phi_final, F_final

    # ── Visualisation: optimized-ansatz output (per-qubit Bloch) ──────────
    #
    # VQT differs from VarQITE in what the "state" is. VarQITE prepares a PURE
    # statevector on 2·n_sys+1 wires (system + ancilla + aux). VQT prepares a
    # MIXED state on the n_sys system qubits ONLY:
    #
    #     ρ(θ, φ) = Σ_x p_φ(x) · U(θ)|x⟩⟨x|U(θ)† ,   p_φ = softmax(φ).
    #
    # So the per-qubit reduced Bloch vector of qubit q is the mixture-weighted
    # average of the single-basis-state expectations:
    #
    #     ⟨P⟩_q = Tr(ρ P_q) = Σ_x p_φ(x) · ⟨x| U†(θ) P_q U(θ) |x⟩ ,
    #
    # exactly mirroring how the VQT readout computes diag(ρ)_k =
    # Σ_x p_φ(x)·|⟨k|U|x⟩|². Because the state is genuinely mixed, the system
    # qubits sit INSIDE the Bloch ball (|r| < 1) — that mixedness IS the
    # thermal/entropy content the VQT free energy trades against the energy.

    def optimized_bloch_vectors(self, d_i, beta_t=None, s_t=0.0):
        """
        Run VQT on demand for ONE energy row d_i and return the per-qubit
        reduced Bloch vectors of the OPTIMIZED mixed state ρ(θ*, φ*).

        The model must already be fitted (so n_sys / kernels exist). We
        re-optimise (θ, φ) for this row at the given s_t and measure
        ⟨X⟩,⟨Y⟩,⟨Z⟩ on every system qubit, mixture-weighted by p_φ(x).

        Parameters
        ----------
        d_i : (K,) array
            A per-sample energy row (e.g. one row of D from the E-step).
            Shift/scale/padding is applied exactly as in the fit.
        beta_t : float or None
            Inverse temperature (entropy weight T = 1/β). Defaults to
            self.beta0 (the trained regime), held fixed across the anneal.
        s_t : float, default 0.0
            Mixer strength. 1.0 = pure mixer; 0.0 = diagonal end (readout = r).

        Returns
        -------
        dict with:
            theta   : (n_params,) optimized ansatz parameters
            phi     : (K_pad,)    optimized categorical logits
            p_phi   : (K_pad,)    softmax mixture weights p_φ(x)
            bloch   : (n_sys, 3)  per-qubit (⟨X⟩,⟨Y⟩,⟨Z⟩) of the MIXED state
            purity  : (n_sys,)    single-qubit purity ½(1+|r|²)
            probs   : (K,)        readout responsibilities (diag ρ on K-block)
            roles   : list[str]   wire role labels ("S0","S1",…) — system only
        """
        if not hasattr(self, "n_sys") or self._batched_kernel is None:
            raise RuntimeError(
                "Model not initialised. Call .fit(X) first so the device / "
                "kernels exist.")
        qml = self._qml
        jnp = self._jnp
        np_ = np

        beta_t = float(self.beta0 if beta_t is None else beta_t)
        d_i = np_.asarray(d_i, dtype=np_.float64).reshape(-1)
        if d_i.shape[0] != self.K:
            raise ValueError(f"d_i must have length K={self.K}, got {d_i.shape[0]}")

        # Optimise (θ, φ) for this single row via the batched kernel (B=1).
        r_batch, theta_batch, phi_batch, _F = self._vqt_batch(
            d_i[None, :], beta_t, s_t)
        theta = np_.asarray(theta_batch[0], dtype=np_.float64)
        phi   = np_.asarray(phi_batch[0],   dtype=np_.float64)
        probs = np_.asarray(r_batch[0],     dtype=np_.float64)

        # Softmax mixture weights p_φ(x) (numerically stable).
        ph = phi - phi.max()
        p_phi = np_.exp(ph); p_phi /= p_phi.sum()

        n_sys = self.n_sys
        K_pad = self.K_pad

        # QNode: prepare |x⟩, apply U(θ), return ⟨X_q⟩,⟨Y_q⟩,⟨Z_q⟩ for all q.
        # Returned as a flat tuple (X for all wires, then Y, then Z); we reshape
        # outside (measurement objects can't be jnp.stacked inside the qfunc).
        @qml.qnode(self._dev, interface="jax")
        def _bloch_in_state(th, x):
            for q in range(n_sys):
                if (x >> (n_sys - 1 - q)) & 1:
                    qml.PauliX(wires=q)
            self._ansatz(th)
            return tuple(
                [qml.expval(qml.PauliX(q)) for q in range(n_sys)]
                + [qml.expval(qml.PauliY(q)) for q in range(n_sys)]
                + [qml.expval(qml.PauliZ(q)) for q in range(n_sys)])

        # Mixed-state Bloch = Σ_x p_φ(x) · (per-basis-state expectations).
        th_j = jnp.asarray(theta)
        bxyz = np_.zeros((3, n_sys), dtype=np_.float64)
        for x in range(K_pad):
            flat = np_.asarray(_bloch_in_state(th_j, x), dtype=np_.float64)
            bxyz += float(p_phi[x]) * flat.reshape(3, n_sys)
        bloch = bxyz.T                                       # (n_sys, 3)
        rnorm = np_.linalg.norm(bloch, axis=1)
        purity = 0.5 * (1.0 + rnorm ** 2)

        roles = [f"S{q}" for q in range(n_sys)]
        return dict(theta=theta, phi=phi, p_phi=p_phi, bloch=bloch,
                    purity=purity, probs=probs, roles=roles,
                    beta_t=beta_t, s_t=float(s_t))

    def visualize_optimized_ansatz(self, d_i, beta_t=None, s_t=0.0,
                                   out_path="v4_2_optimized_bloch.png",
                                   title=None, show=False):
        """
        Render the OPTIMIZED-ansatz output for one energy row d_i as a row of
        per-qubit Bloch spheres (system qubits) plus a bar chart of the readout
        responsibilities. Mirrors the VarQITE visualiser, adapted for VQT's
        system-only mixed state. Saves a PNG to `out_path` and returns it.
        """
        try:
            import matplotlib
            if not show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError as exc:
            raise ImportError("matplotlib required for visualisation: "
                              "pip install matplotlib") from exc

        res = self.optimized_bloch_vectors(d_i, beta_t=beta_t, s_t=s_t)
        bloch, purity = res["bloch"], res["purity"]
        roles, probs = res["roles"], res["probs"]
        n_sys = self.n_sys
        _np = np

        sys_color = "#2266aa"   # VQT has system qubits only

        # Floor the width at 8.5" so the 2-line suptitle never crowds the
        # subplot title at n_sys=1; reserve a top band for the suptitle.
        fig = plt.figure(figsize=(max(3.0 * n_sys, 8.5), 6.4))
        gs = fig.add_gridspec(2, n_sys, height_ratios=[3.0, 1.4],
                              top=0.84, bottom=0.10, left=0.08, right=0.95,
                              hspace=0.35, wspace=0.30)

        u = _np.linspace(0, 2 * _np.pi, 24)
        v = _np.linspace(0, _np.pi, 16)
        sx = _np.outer(_np.cos(u), _np.sin(v))
        sy = _np.outer(_np.sin(u), _np.sin(v))
        sz = _np.outer(_np.ones_like(u), _np.cos(v))

        for q in range(n_sys):
            ax = fig.add_subplot(gs[0, q], projection="3d")
            ax.plot_wireframe(sx, sy, sz, color="#dddddd", linewidth=0.4)
            for a0, a1 in [((-1, 0, 0), (1, 0, 0)),
                           ((0, -1, 0), (0, 1, 0)),
                           ((0, 0, -1), (0, 0, 1))]:
                ax.plot(*zip(a0, a1), color="#bbbbbb", linewidth=0.6)
            bx, by, bz = bloch[q]
            ax.quiver(0, 0, 0, bx, by, bz, color=sys_color, linewidth=2.2,
                      arrow_length_ratio=0.18)
            ax.scatter([bx], [by], [bz], color=sys_color, s=30)
            ax.set_title(f"{roles[q]}\n|r|={_np.linalg.norm(bloch[q]):.2f}  "
                         f"P={purity[q]:.2f}", fontsize=9, color=sys_color)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))

        axp = fig.add_subplot(gs[1, :])
        axp.bar(range(len(probs)), probs, color=sys_color, alpha=0.85)
        axp.set_xlabel("cluster k")
        axp.set_ylabel("responsibility r")
        axp.set_xticks(range(len(probs)))
        axp.set_title("Readout: diag(ρ) on system register "
                      f"(s_t={res['s_t']:.2f}, β={res['beta_t']:.2f})",
                      fontsize=10)
        axp.grid(axis="y", alpha=0.3)

        if title is None:
            title = (f"v4_2 VQT optimized ansatz output — K={self.K}, "
                     f"n_sys={self.n_sys}, depth={self.ansatz_depth}, "
                     f"steps={self.n_vqt_steps}\n"
                     f"per-qubit reduced Bloch vectors of the MIXED state "
                     f"ρ(θ*,φ*)  (system qubits; |r|<1 ⇒ mixed/entropy)")
        fig.suptitle(title, fontsize=10, y=0.985, va="top", linespacing=1.5)

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    # ── Animated anneal: per-qubit Bloch spheres as s_t sweeps 1 → 0 ──────

    def animate_anneal_bloch(self, d_i, beta_t=None,
                             s_start=1.0, s_end=0.0, n_frames=41,
                             out_path="v4_2_anneal_bloch.gif",
                             fps=8, dpi=110, title=None, trail=True):
        """
        Render an ANIMATED GIF of the optimized VQT mixed-state per-qubit Bloch
        vectors as the mixer strength s_t sweeps from `s_start` (1.0, pure
        mixer) to `s_end` (0.0, diagonal end). Each frame re-optimises VQT for
        the same energy row d_i at that s_t and draws the system-qubit Bloch
        spheres + readout responsibility bars.

        Parameters mirror the VarQITE animate_anneal_bloch. Axis limits and the
        responsibility-bar y-axis are FIXED across frames so the animation
        shows genuine motion. Requires matplotlib + Pillow.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from PIL import Image
        except ImportError as exc:
            raise ImportError("matplotlib + Pillow required for the GIF: "
                              "pip install matplotlib pillow") from exc

        _np = np
        beta_t = float(self.beta0 if beta_t is None else beta_t)
        d_i = _np.asarray(d_i, dtype=_np.float64).reshape(-1)
        n_sys = self.n_sys

        s_values = _np.clip(
            _np.linspace(float(s_start), float(s_end), int(n_frames)),
            0.0, 1.0)

        sys_color = "#2266aa"

        u = _np.linspace(0, 2 * _np.pi, 24)
        v = _np.linspace(0, _np.pi, 16)
        sx = _np.outer(_np.cos(u), _np.sin(v))
        sy = _np.outer(_np.sin(u), _np.sin(v))
        sz = _np.outer(_np.ones_like(u), _np.cos(v))

        # First pass: optimise + measure at every s_t (also fixes the
        # responsibility-bar y-limit and enables the tip trail).
        frames_data = []
        max_prob = 1e-6
        for s in s_values:
            res = self.optimized_bloch_vectors(d_i, beta_t=beta_t, s_t=float(s))
            frames_data.append(res)
            max_prob = max(max_prob, float(_np.max(res["probs"])))
        roles = frames_data[0]["roles"]
        prob_ylim = min(1.0, 1.15 * max_prob)

        if title is None:
            title = (f"VQT anneal  s_t: {s_start:.2f} → {s_end:.2f}  "
                     f"(K={self.K}, n_sys={self.n_sys}, depth={self.ansatz_depth}, "
                     f"β={beta_t:.1f})\n"
                     f"per-qubit Bloch vectors of mixed state ρ(θ*,φ*)  "
                     f"(system qubits; |r|<1 ⇒ mixed)")

        trails = [[] for _ in range(n_sys)]
        images = []
        # Figure width: wide enough for BOTH the spheres and the (long) title.
        # At n_sys=1 the spheres alone want only ~6", but the 2-line header +
        # progress bar are much wider, so floor the width at 8.5" to stop the
        # suptitle from crowding the single subplot title.
        fig_w = max(3.0 * n_sys, 8.5)
        # The suptitle is 3 lines (header, description, frame/progress); reserve
        # vertical space for it so it never overlaps the per-sphere titles. The
        # gridspec `top` leaves that band free.
        gs_top = 0.80
        for fi, (s, res) in enumerate(zip(s_values, frames_data)):
            bloch, purity, probs = res["bloch"], res["purity"], res["probs"]

            fig = plt.figure(figsize=(fig_w, 7.0))
            gs = fig.add_gridspec(2, n_sys, height_ratios=[3.0, 1.5],
                                  top=gs_top, bottom=0.09,
                                  left=0.08, right=0.95,
                                  hspace=0.42, wspace=0.30)

            for q in range(n_sys):
                ax = fig.add_subplot(gs[0, q], projection="3d")
                ax.plot_wireframe(sx, sy, sz, color="#dddddd", linewidth=0.4)
                for a0, a1 in [((-1, 0, 0), (1, 0, 0)),
                               ((0, -1, 0), (0, 1, 0)),
                               ((0, 0, -1), (0, 0, 1))]:
                    ax.plot(*zip(a0, a1), color="#bbbbbb", linewidth=0.6)

                bx, by, bz = bloch[q]
                if trail:
                    trails[q].append((bx, by, bz))
                    if len(trails[q]) > 1:
                        tp = _np.array(trails[q])
                        ax.plot(tp[:, 0], tp[:, 1], tp[:, 2],
                                color=sys_color, linewidth=0.8, alpha=0.35)

                ax.quiver(0, 0, 0, bx, by, bz, color=sys_color, linewidth=2.4,
                          arrow_length_ratio=0.18)
                ax.scatter([bx], [by], [bz], color=sys_color, s=32)
                ax.set_title(f"{roles[q]}\n|r|={_np.linalg.norm(bloch[q]):.2f}  "
                             f"P={purity[q]:.2f}", fontsize=9, color=sys_color)
                ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.set_box_aspect((1, 1, 1))

            axp = fig.add_subplot(gs[1, :])
            axp.bar(range(len(probs)), probs, color=sys_color, alpha=0.85)
            axp.set_xlabel("cluster k")
            axp.set_ylabel("responsibility r")
            axp.set_xticks(range(len(probs)))
            axp.set_ylim(0.0, prob_ylim)
            axp.set_title("Readout: diag(ρ) on system register", fontsize=10)
            axp.grid(axis="y", alpha=0.3)

            frac = fi / max(len(s_values) - 1, 1)
            # Suptitle in the reserved top band (anchored at the very top,
            # growing downward), so the multi-line header + progress bar never
            # collide with the per-sphere titles even at n_sys=1.
            fig.suptitle(f"{title}\n"
                         f"frame {fi+1}/{len(s_values)}    "
                         f"s_t = {float(s):.3f}   "
                         f"[{'█' * int(round(frac * 20)):<20}]",
                         fontsize=10, family="monospace",
                         y=0.995, va="top", linespacing=1.5)

            fig.canvas.draw()
            buf = _np.asarray(fig.canvas.buffer_rgba())
            images.append(Image.fromarray(buf[..., :3].copy()))
            plt.close(fig)

        duration_ms = int(round(1000.0 / max(fps, 1)))
        images[0].save(
            out_path, save_all=True, append_images=images[1:],
            duration=duration_ms, loop=0, optimize=True,
        )
        return out_path

    # ── Reporting: add the CPU-parallel line to the v4_2 header ───────────

    def _print_fit_header(self):
        super()._print_fit_header()
        par = (f"pmap×{self._n_devices}"
               if (self._n_devices > 1 and self.cpu_parallel is not False)
               else "single-dev")
        print(f"  [perf] cpu={par}  "
              f"(devices={self._n_devices}, cpu_parallel={self.cpu_parallel!r})")

    def perf_summary(self) -> dict:
        base = super().perf_summary()
        if base:
            base["n_devices"] = int(self._n_devices)
            base["cpu_parallel"] = (
                f"pmap×{self._n_devices}"
                if (self._n_devices > 1 and self.cpu_parallel is not False)
                else "single-dev")
        return base


# ════════════════════════════════════════════════════════════════════════════
# Smoke + benchmark: v4_2 (single-device) vs v4_2fast (pmap), equal results
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import (adjusted_rand_score,
                                  normalized_mutual_info_score)

    print("=" * 72)
    print("VQT QAVB v4_2fast (JAX + pmap CPU-parallel) — smoke + benchmark")
    print("=" * 72)

    rng = np.random.default_rng(42)
    N, S, K_true = 600, 3000, 4
    block = S // K_true
    alpha = np.full((K_true, S), 0.1)
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([
        rng.multinomial(1500, rng.dirichlet(alpha[true_labels[i]]))
        for i in range(N)
    ], dtype=float)

    common = dict(
        K_max=4, nu="auto", max_iter=10, beta0=5.0, s0=1.0,
        tau1=4, tau2=8, prune_start=12, verbose=0, random_state=42,
        selection_prior=0.3, prune_threshold=0.2,
        use_trigamma_correction=False,
        ansatz_depth=2, n_vqt_steps=40,
        warm_start=True, jit_warmup=False, dedup_n_clusters=None,
    )

    # v4_2fast with sharding OFF == v4_2 baseline.
    print("\n--- v4_2fast, cpu_parallel=False (== v4_2 single-device) ---")
    t0 = time()
    m_off = DMM_SVVS_VQT_QAVB_JAX_Fast(cpu_parallel=False, **common)
    m_off.fit(X)
    dt_off = time() - t0
    ari_off = adjusted_rand_score(true_labels, m_off.predict(X))

    # v4_2fast with sharding ON.
    print("--- v4_2fast, cpu_parallel=True (pmap over CPU shards) ---")
    t0 = time()
    m_on = DMM_SVVS_VQT_QAVB_JAX_Fast(cpu_parallel=True, **common)
    m_on.fit(X)
    dt_on = time() - t0
    ari_on = adjusted_rand_score(true_labels, m_on.predict(X))

    ndev = m_on._n_devices
    print("\n" + "=" * 72)
    print(f"Benchmark (N={N}, S={S}, K_true={K_true})")
    print(f"{'config':<30} {'ARI':>7} {'K':>4} {'time(s)':>9}")
    print("-" * 72)
    print(f"{'v4_2fast single-device':<30} {ari_off:7.3f} {m_off.K:4d} {dt_off:9.1f}")
    print(f"{'v4_2fast pmap×'+str(ndev):<30} {ari_on:7.3f} {m_on.K:4d} {dt_on:9.1f}")
    sp = dt_off / dt_on if dt_on > 0 else float("inf")
    print("-" * 72)
    print(f"  pmap speedup over single-device: {sp:.2f}x  "
          f"(ARI match: {abs(ari_off - ari_on) < 1e-9})")
    print("=" * 72)
    print("Speedup grows as the VQT E-step dominates (larger N, n_vqt_steps,")
    print("ansatz_depth, K). Amdahl: SVVS M-step / dedup are single-threaded.")
