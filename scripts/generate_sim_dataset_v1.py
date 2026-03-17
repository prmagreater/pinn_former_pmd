#!/usr/bin/env python3
"""
Generate PMD simulation dataset v1 (surface_gt + deflect_obs + cmm_points + calib).

Examples:
  python scripts/generate_sim_dataset_v1.py --out_root ./sim_dataset_v1
  python scripts/generate_sim_dataset_v1.py --out_root ./sim_dataset_v1 --num_samples 30 --seed 42 --include_design 1
  python scripts/generate_sim_dataset_v1.py --out_root ./sim_dataset_v1 --H 384 --W 384 --R 0.025 --family_mix "A:0.5,B:0.2,C:0.3"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_manifest(repo_root: Path) -> Dict:
    with (repo_root / "dataset_manifest_v1.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_family_mix(text: str) -> Dict[str, float]:
    pairs = [x.strip() for x in text.split(",") if x.strip()]
    out = {"A": 0.0, "B": 0.0, "C": 0.0}
    for p in pairs:
        k, v = p.split(":")
        k = k.strip().upper()
        if k not in out:
            raise ValueError(f"Unsupported family key: {k}")
        out[k] = float(v)
    s = sum(out.values())
    if s <= 0:
        raise ValueError("family_mix sum must be > 0")
    for k in out:
        out[k] /= s
    return out


def allocate_families(num_samples: int, mix: Dict[str, float]) -> list[str]:
    keys = ["A", "B", "C"]
    raw = np.array([mix[k] * num_samples for k in keys], dtype=float)
    base = np.floor(raw).astype(int)
    remain = num_samples - int(base.sum())
    frac_order = np.argsort(-(raw - base))
    for i in range(remain):
        base[frac_order[i]] += 1
    fams = []
    for k, n in zip(keys, base.tolist()):
        fams.extend([k] * n)
    return fams


def make_grid(H: int, W: int, R: float) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    x0 = -R
    y0 = -R
    dx = (2.0 * R) / (W - 1)
    dy = (2.0 * R) / (H - 1)
    xs = x0 + np.arange(W) * dx
    ys = y0 + np.arange(H) * dy
    X, Y = np.meshgrid(xs, ys)
    aperture_mask = (X * X + Y * Y) <= (R * R)
    return X, Y, x0, y0, dx, dy, aperture_mask


def robust_scale_to_rms(delta: np.ndarray, mask: np.ndarray, target_rms: float) -> np.ndarray:
    cur = np.sqrt(np.mean((delta[mask]) ** 2)) + 1e-18
    return delta * (target_rms / cur)


def gen_design(family: str, X: np.ndarray, Y: np.ndarray, R: float, fam_cfg: Dict, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    xn = X / R
    yn = Y / R
    if family == "A":
        # Mild freeform polynomial-like design.
        c1 = rng.uniform(2e-4, 9e-4)
        c2 = rng.uniform(-3e-4, 3e-4)
        c3 = rng.uniform(-3e-4, 3e-4)
        c4 = rng.uniform(-2e-4, 2e-4)
        z = c1 * (0.7 * xn**2 + 0.3 * yn**2) + c2 * xn * yn + c3 * xn**3 + c4 * yn**3
    elif family == "B":
        # Stronger toroid/saddle challenge shape.
        a = rng.uniform(5e-4, 15e-4)
        b = rng.uniform(-10e-4, -2e-4)
        c = rng.uniform(-5e-4, 5e-4)
        d = rng.uniform(-3e-4, 3e-4)
        z = a * xn**2 + b * yn**2 + c * xn * yn + d * (xn**3 - yn**3)
    elif family == "C":
        # A-like base + mild deterministic multiscale term in design.
        c1 = rng.uniform(2e-4, 8e-4)
        c2 = rng.uniform(-3e-4, 3e-4)
        f = rng.uniform(2.0, 4.0)
        z = c1 * (xn**2 + 0.6 * yn**2) + c2 * xn * yn + 8e-5 * np.sin(2 * np.pi * f * xn)
    else:
        raise ValueError(f"Unknown family: {family}")

    pv_rng = fam_cfg["target_pv_sag_m_range"]
    target_pv = rng.uniform(pv_rng[0], pv_rng[1])
    cur_pv = float(np.nanmax(z) - np.nanmin(z)) + 1e-18
    z *= target_pv / cur_pv
    return z, {"target_pv": target_pv, "coeff_seed": int(rng.integers(0, 2**31 - 1))}


def gen_LF(X: np.ndarray, Y: np.ndarray, R: float, cfg: Dict, rng: np.random.Generator, aperture_mask: np.ndarray) -> np.ndarray:
    if not cfg["enabled"]:
        return np.zeros_like(X)
    terms_lo, terms_hi = cfg["terms_range"]
    n_terms = int(rng.integers(terms_lo, terms_hi + 1))
    basis = []
    xn, yn = X / R, Y / R
    for i in range(4):
        for j in range(4 - i):
            if i + j == 0:
                continue
            basis.append((xn**i) * (yn**j))
    rng.shuffle(basis)
    delta = np.zeros_like(X)
    for k in range(n_terms):
        delta += rng.normal(0.0, 1.0) * basis[k]
    rms_lo, rms_hi = cfg["rms_m_range"]
    target_rms = rng.uniform(rms_lo, rms_hi)
    return robust_scale_to_rms(delta, aperture_mask, target_rms)


def gen_MSF(X: np.ndarray, Y: np.ndarray, cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    if not cfg["enabled"]:
        return np.zeros_like(X)
    K_lo, K_hi = cfg["sinusoids_K_range"]
    K = int(rng.integers(K_lo, K_hi + 1))
    amp_lo, amp_hi = cfg["amplitude_m_range"]
    wl_lo, wl_hi = cfg["wavelength_m_range"]
    z = np.zeros_like(X)
    for _ in range(K):
        amp = rng.uniform(amp_lo, amp_hi) * rng.choice([-1.0, 1.0])
        wl = rng.uniform(wl_lo, wl_hi)
        theta = rng.uniform(0, 2 * np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        kx = np.cos(theta) * 2 * np.pi / wl
        ky = np.sin(theta) * 2 * np.pi / wl
        z += amp * np.sin(kx * X + ky * Y + phase)

    if cfg.get("random_field_enabled", False):
        rf_lo, rf_hi = cfg["random_field_rms_m_range"]
        rf = rng.normal(0.0, 1.0, size=X.shape)
        rf = rf - rf.mean()
        rf = rf / (rf.std() + 1e-18)
        z += rf * rng.uniform(rf_lo, rf_hi)
    return z


def gen_defect(X: np.ndarray, Y: np.ndarray, cfg: Dict, rng: np.random.Generator) -> Tuple[np.ndarray, str]:
    if not cfg["enabled"]:
        return np.zeros_like(X), "none"
    types = cfg["types"]
    p_pit = float(types["gaussian_pit"]["prob"])
    p_scratch = float(types["scratch"]["prob"])
    s = p_pit + p_scratch
    p_pit /= s
    choose_pit = rng.random() < p_pit

    if choose_pit:
        c = types["gaussian_pit"]
        A = rng.uniform(*c["A_m_abs_range"]) * -1.0
        sigma = rng.uniform(*c["sigma_m_range"])
        rr = c["center_radius_m_max"] * np.sqrt(rng.random())
        tt = rng.uniform(0, 2 * np.pi)
        x0, y0 = rr * np.cos(tt), rr * np.sin(tt)
        z = A * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
        return z, "gaussian_pit"

    c = types["scratch"]
    A = rng.uniform(*c["A_m_abs_range"]) * -1.0
    width = rng.uniform(*c["width_m_range"])
    length = rng.uniform(*c["length_m_range"])
    rr = c["center_radius_m_max"] * np.sqrt(rng.random())
    tt = rng.uniform(0, 2 * np.pi)
    x0, y0 = rr * np.cos(tt), rr * np.sin(tt)
    ang = rng.uniform(0, 2 * np.pi)
    ca, sa = np.cos(ang), np.sin(ang)
    xr = ca * (X - x0) + sa * (Y - y0)
    yr = -sa * (X - x0) + ca * (Y - y0)
    envelope = np.exp(-(yr**2) / (2 * width**2))
    gate = (np.abs(xr) <= (length / 2)).astype(float)
    z = A * envelope * gate
    return z, "scratch"


def compute_slope(z: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dzdy, dzdx = np.gradient(z, dy, dx)
    s = np.sqrt(dzdx**2 + dzdy**2)
    return dzdx, dzdy, s


def enforce_physical_constraints(z_design: np.ndarray, z_lf: np.ndarray, z_msf: np.ndarray, z_defect: np.ndarray,
                                 aperture_mask: np.ndarray, dx: float, dy: float,
                                 slope_limit: float, pv_range: Tuple[float, float]) -> np.ndarray:
    alpha = 1.0
    z = z_design + alpha * (z_lf + z_msf + z_defect)
    _, _, s = compute_slope(z, dx, dy)
    maxs = float(np.max(s[aperture_mask]))
    if maxs > slope_limit:
        alpha *= slope_limit / (maxs + 1e-18)

    z = z_design + alpha * (z_lf + z_msf + z_defect)
    pv = float(np.max(z[aperture_mask]) - np.min(z[aperture_mask]))
    if pv > pv_range[1]:
        scale = pv_range[1] / (pv + 1e-18)
        z *= scale
    elif pv < pv_range[0]:
        scale = pv_range[0] / (pv + 1e-18)
        z *= scale

    _, _, s2 = compute_slope(z, dx, dy)
    maxs2 = float(np.max(s2[aperture_mask]))
    if maxs2 > slope_limit:
        z *= slope_limit / (maxs2 + 1e-18)
    return z


def forward_uv(z: np.ndarray, X: np.ndarray, Y: np.ndarray, aperture_mask: np.ndarray, Cz: float, Zs: float,
               screen_w: float, screen_h: float, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, q, _ = compute_slope(z, dx, dy)
    denom = np.sqrt(1.0 + p * p + q * q)
    n = np.stack((-p / denom, -q / denom, 1.0 / denom), axis=-1)

    P = np.stack((X, Y, z), axis=-1)
    C = np.array([0.0, 0.0, Cz], dtype=np.float64)
    d_in = P - C
    d_in = d_in / (np.linalg.norm(d_in, axis=-1, keepdims=True) + 1e-18)
    dot = np.sum(d_in * n, axis=-1, keepdims=True)
    d_out = d_in - 2.0 * dot * n

    dz = d_out[..., 2]
    valid_dir = np.abs(dz) > 1e-12
    t = (Zs - z) / (dz + 1e-18)
    valid_t = t > 0

    U = X + t * d_out[..., 0]
    V = Y + t * d_out[..., 1]
    in_screen = (np.abs(U) <= screen_w / 2.0) & (np.abs(V) <= screen_h / 2.0)

    # Reflection direction roughly must point from mirror towards screen plane.
    physically_reasonable = (dz * np.sign(Zs - z)) > 0
    valid = aperture_mask & valid_dir & valid_t & in_screen & physically_reasonable

    U = np.where(valid, U, np.nan)
    V = np.where(valid, V, np.nan)
    conf = valid.astype(np.float32)
    return U, V, conf


def add_poly_distortion(X: np.ndarray, Y: np.ndarray, coef_scale: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # degree-2 polynomial terms: 1, x, y, x^2, x*y, y^2
    x = X
    y = Y
    feats = [np.ones_like(x), x, y, x * x, x * y, y * y]
    cu = rng.normal(0.0, coef_scale, size=len(feats))
    cv = rng.normal(0.0, coef_scale, size=len(feats))
    du = np.zeros_like(x)
    dv = np.zeros_like(y)
    for i, f in enumerate(feats):
        du += cu[i] * f
        dv += cv[i] * f
    return du, dv


def apply_blob_field(X: np.ndarray, Y: np.ndarray, num_blobs: int, sigma_rng: Tuple[float, float],
                     rng: np.random.Generator) -> np.ndarray:
    field = np.zeros_like(X)
    for _ in range(num_blobs):
        sigma = rng.uniform(*sigma_rng)
        rr = 0.02 * np.sqrt(rng.random())
        tt = rng.uniform(0, 2 * np.pi)
        x0, y0 = rr * np.cos(tt), rr * np.sin(tt)
        amp = rng.uniform(0.6, 1.0)
        field += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
    mx = np.max(field)
    return field / (mx + 1e-18)


def add_obs_noise(U: np.ndarray, V: np.ndarray, conf_phys: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  obs_cfg: Dict, non_geo_cfg: Dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
    Uo = U.copy()
    Vo = V.copy()
    valid = conf_phys > 0

    uvn = obs_cfg["uv_gaussian_noise"]
    sigma_u = rng.uniform(*uvn["sigma_u_m_range"]) if uvn["enabled"] else 0.0
    sigma_v = rng.uniform(*uvn["sigma_v_m_range"]) if uvn["enabled"] else 0.0
    Uo[valid] = Uo[valid] + rng.normal(0.0, sigma_u, size=np.count_nonzero(valid))
    Vo[valid] = Vo[valid] + rng.normal(0.0, sigma_v, size=np.count_nonzero(valid))

    uvl = obs_cfg["uv_lowfreq_distortion"]
    coef_scale = rng.uniform(*uvl["coef_scale_m_range"]) if uvl["enabled"] else 0.0
    if uvl["enabled"] and coef_scale > 0:
        du, dv = add_poly_distortion(X, Y, coef_scale, rng)
        Uo[valid] += du[valid]
        Vo[valid] += dv[valid]

    noise_boost = np.zeros_like(U)
    ncfg = non_geo_cfg["uv_noise_boost"]
    if non_geo_cfg["enabled"] and (rng.random() < ncfg["prob"]):
        nb = int(rng.integers(ncfg["num_blobs_range"][0], ncfg["num_blobs_range"][1] + 1))
        noise_boost = apply_blob_field(X, Y, nb, tuple(ncfg["blob_sigma_m_range"]), rng)
        mult = rng.uniform(*ncfg["noise_scale_multiplier_range"])
        su = sigma_u * (1.0 + (mult - 1.0) * noise_boost)
        sv = sigma_v * (1.0 + (mult - 1.0) * noise_boost)
        Uo[valid] += rng.normal(0.0, su[valid])
        Vo[valid] += rng.normal(0.0, sv[valid])

    noise_summary = {
        "sigma_u": float(sigma_u),
        "sigma_v": float(sigma_v),
        "coef_scale": float(coef_scale),
    }
    return Uo, Vo, noise_summary, noise_boost


def build_conf(conf_phys: np.ndarray, aperture_mask: np.ndarray, slope: np.ndarray, X: np.ndarray, Y: np.ndarray,
               obs_cfg: Dict, non_geo_cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    conf = conf_phys.astype(np.float64).copy()
    cm = obs_cfg["conf_masking"]
    if cm["enabled"]:
        if cm["edge_falloff"]["enabled"]:
            power = rng.uniform(*cm["edge_falloff"]["power_range"])
            r = np.sqrt(X * X + Y * Y)
            rmax = np.max(r[aperture_mask]) + 1e-18
            edge = np.clip(1.0 - (r / rmax) ** power, 0.0, 1.0)
            conf *= edge
        if cm["high_slope_penalty"]["enabled"]:
            s0 = rng.uniform(*cm["high_slope_penalty"]["slope0_range"])
            conf *= np.exp(-((slope / (s0 + 1e-18)) ** 2))
        if cm["random_drop"]["enabled"]:
            p = rng.uniform(*cm["random_drop"]["drop_prob_range"])
            drop = rng.random(size=conf.shape) < p
            conf[drop] = 0.0

    cdrop = non_geo_cfg["conf_drop"]
    if non_geo_cfg["enabled"] and (rng.random() < cdrop["prob"]):
        nb = int(rng.integers(cdrop["num_blobs_range"][0], cdrop["num_blobs_range"][1] + 1))
        fld = apply_blob_field(X, Y, nb, tuple(cdrop["blob_sigma_m_range"]), rng)
        min_scale = rng.uniform(*cdrop["min_conf_scale_range"])
        scale = 1.0 - (1.0 - min_scale) * fld
        conf *= scale

    conf = np.where(aperture_mask, conf, 0.0)
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
    return conf


def interp_bilinear(z: np.ndarray, xq: np.ndarray, yq: np.ndarray, x0: float, y0: float, dx: float, dy: float) -> np.ndarray:
    fx = (xq - x0) / dx
    fy = (yq - y0) / dy
    x0i = np.floor(fx).astype(int)
    y0i = np.floor(fy).astype(int)
    x1i = np.clip(x0i + 1, 0, z.shape[1] - 1)
    y1i = np.clip(y0i + 1, 0, z.shape[0] - 1)
    x0i = np.clip(x0i, 0, z.shape[1] - 1)
    y0i = np.clip(y0i, 0, z.shape[0] - 1)

    wx = fx - x0i
    wy = fy - y0i
    z00 = z[y0i, x0i]
    z10 = z[y0i, x1i]
    z01 = z[y1i, x0i]
    z11 = z[y1i, x1i]
    z0 = z00 * (1 - wx) + z10 * wx
    z1 = z01 * (1 - wx) + z11 * wx
    return z0 * (1 - wy) + z1 * wy


def sample_cmm(z_true: np.ndarray, N: int, R: float, x0: float, y0: float, dx: float, dy: float,
               cmm_cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    rr = R * np.sqrt(rng.random(N))
    tt = rng.uniform(0, 2 * np.pi, size=N)
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    z = interp_bilinear(z_true, x, y, x0, y0, dx, dy)
    zn = rng.uniform(*cmm_cfg["z_noise_m_range"])
    z += rng.normal(0.0, zn, size=N)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts


def build_calib(manifest: Dict, H: int, W: int, R: float, x0: float, y0: float, dx: float, dy: float,
                family: str, design_id: str, design_params: Dict, noise_summary: Dict,
                defect_type: str, valid_ratio: float, sample_seed: int) -> Dict:
    Cz = manifest["extrinsics_v1_placeholder"]["T_WC_translation_m"][2]
    Zs = manifest["extrinsics_v1_placeholder"]["T_WS_translation_m"][2]

    def T(tx: float, ty: float, tz: float):
        return [[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]]

    return {
        "units": manifest["units"],
        "mirror_grid": {
            "H": H,
            "W": W,
            "x0_m": x0,
            "y0_m": y0,
            "dx_m": dx,
            "dy_m": dy,
        },
        "aperture": {"type": "circle", "radius_m": R},
        "camera_intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": W / 2.0, "cy": H / 2.0},
        "extrinsics": {
            "T_WM": T(0.0, 0.0, 0.0),
            "T_WC": T(0.0, 0.0, Cz),
            "T_WS": T(0.0, 0.0, Zs),
        },
        "screen": manifest["screen"],
        "design": {
            "surface_family": family,
            "design_id": design_id,
            "design_params": design_params,
            "design_seed": sample_seed,
        },
        "notes": "v1 synthetic dataset; no GT slopes/normals/defect masks saved.",
        "meta_summary": {
            "surface_family": family,
            "design_id": design_id,
            "noise_config": {
                "sigma_u": noise_summary["sigma_u"],
                "sigma_v": noise_summary["sigma_v"],
                "valid_ratio": valid_ratio,
            },
            "defect_type": defect_type,
            "valid_ratio": valid_ratio,
        },
    }


def anti_leakage_asserts(sample_dir: Path, H: int, W: int, include_design: bool):
    bad_patterns = ["p_gt", "q_gt", "n_gt", "defect_mask", "forward_intermediates"]
    names = [p.name for p in sample_dir.iterdir()]
    for n in names:
        for pat in bad_patterns:
            assert pat not in n, f"Leakage file found: {n}"

    d = np.load(sample_dir / "deflect_obs.npy")
    assert d.shape == (H, W, 3), f"deflect_obs shape mismatch: {d.shape}"
    assert np.nanmin(d[..., 2]) >= 0.0 and np.nanmax(d[..., 2]) <= 1.0, "conf out of [0,1]"

    c = np.load(sample_dir / "cmm_points.npy")
    assert c.ndim == 2 and c.shape[1] == 3, f"cmm_points shape mismatch: {c.shape}"

    if include_design:
        z = np.load(sample_dir / "design_surface.npy")
        assert z.shape == (H, W), f"design_surface shape mismatch: {z.shape}"


def save_sample(sample_dir: Path, z_true: np.ndarray, deflect_obs: np.ndarray,
                cmm_points: np.ndarray, calib: Dict, z_design: np.ndarray | None):
    sample_dir.mkdir(parents=True, exist_ok=False)
    np.save(sample_dir / "surface_gt.npy", z_true.astype(np.float32))
    np.save(sample_dir / "deflect_obs.npy", deflect_obs.astype(np.float32))
    np.save(sample_dir / "cmm_points.npy", cmm_points.astype(np.float32))
    if z_design is not None:
        np.save(sample_dir / "design_surface.npy", z_design.astype(np.float32))
    with (sample_dir / "calib.json").open("w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate simulation dataset v1 for PMD+CMM")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include_design", type=int, choices=[0, 1], default=1)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--R", type=float, default=None)
    parser.add_argument("--family_mix", type=str, default="A:0.4,B:0.2,C:0.4")
    parser.add_argument("--cmm_n_default", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(repo_root)

    H = args.H if args.H is not None else int(manifest["grid"]["H"])
    W = args.W if args.W is not None else int(manifest["grid"]["W"])
    R = args.R if args.R is not None else float(manifest["grid"]["radius_m"])
    cmm_n = args.cmm_n_default if args.cmm_n_default is not None else int(manifest["cmm_model"]["Ncmm_default"])

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Required by reproducibility section.
    with (out_root / "dataset_manifest_v1.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    family_mix = parse_family_mix(args.family_mix)
    families = allocate_families(args.num_samples, family_mix)
    rng_global = np.random.default_rng(args.seed)
    rng_global.shuffle(families)

    X, Y, x0, y0, dx, dy, aperture_mask = make_grid(H, W, R)

    fam_map = {
        "A": ("A_baseline", "A1"),
        "B": ("B_challenge", "B1"),
        "C": ("C_multiscale", "C1"),
    }

    for idx, family in enumerate(families, start=1):
        sample_seed = args.seed + idx
        rng = np.random.default_rng(sample_seed)

        fam_key, design_id = fam_map[family]
        fam_cfg = manifest["surface_families"][fam_key]

        z_design, design_params = gen_design(family, X, Y, R, fam_cfg, rng)
        z_lf = gen_LF(X, Y, R, manifest["manufacturing_error_model"]["LF_lowfreq"], rng, aperture_mask)
        z_msf = gen_MSF(X, Y, manifest["manufacturing_error_model"]["MSF_midfreq"], rng)
        z_def, defect_type = gen_defect(X, Y, manifest["defect_model"]["geometry_defects"], rng)

        z_true = enforce_physical_constraints(
            z_design, z_lf, z_msf, z_def, aperture_mask,
            dx, dy,
            slope_limit=float(fam_cfg["max_slope_limit"]),
            pv_range=tuple(fam_cfg["target_pv_sag_m_range"]),
        )
        z_true[~aperture_mask] = np.nan

        Cz = float(manifest["extrinsics_v1_placeholder"]["T_WC_translation_m"][2])
        Zs = float(manifest["extrinsics_v1_placeholder"]["T_WS_translation_m"][2])
        sw = float(manifest["screen"]["width_m"])
        sh = float(manifest["screen"]["height_m"])
        U, V, conf_phys = forward_uv(z_true, X, Y, aperture_mask, Cz, Zs, sw, sh, dx, dy)

        Uo, Vo, noise_summary, _ = add_obs_noise(
            U, V, conf_phys, X, Y,
            manifest["observation_noise_model"],
            manifest["defect_model"]["non_geometry_anomalies"],
            rng,
        )

        _, _, slope = compute_slope(np.nan_to_num(z_true, nan=0.0), dx, dy)
        conf = build_conf(
            conf_phys, aperture_mask, slope, X, Y,
            manifest["observation_noise_model"],
            manifest["defect_model"]["non_geometry_anomalies"],
            rng,
        )

        Uo = np.where(conf > 0, Uo, 0.0)
        Vo = np.where(conf > 0, Vo, 0.0)
        deflect_obs = np.stack([Uo, Vo, conf], axis=-1).astype(np.float32)

        cmm_points = sample_cmm(z_true=np.nan_to_num(z_true, nan=0.0), N=cmm_n, R=R,
                                x0=x0, y0=y0, dx=dx, dy=dy,
                                cmm_cfg=manifest["cmm_model"], rng=rng)

        valid_ratio = float(np.mean(conf[aperture_mask]))
        calib = build_calib(
            manifest, H, W, R, x0, y0, dx, dy,
            family=family, design_id=design_id, design_params=design_params,
            noise_summary=noise_summary, defect_type=defect_type,
            valid_ratio=valid_ratio, sample_seed=sample_seed,
        )

        sample_id = f"sample_{family}_{idx:06d}"
        sample_dir = out_root / sample_id
        z_design_save = z_design if args.include_design == 1 else None
        save_sample(sample_dir, z_true, deflect_obs, cmm_points, calib, z_design_save)

        anti_leakage_asserts(sample_dir, H, W, include_design=(args.include_design == 1))

        pv = float(np.nanmax(z_true[aperture_mask]) - np.nanmin(z_true[aperture_mask]))
        max_slope = float(np.max(slope[aperture_mask]))
        slope_limit = float(fam_cfg["max_slope_limit"])
        assert max_slope <= slope_limit + 1e-6, f"slope limit violation: {max_slope} > {slope_limit}"
        pv_lo, pv_hi = fam_cfg["target_pv_sag_m_range"]
        assert pv_lo - 1e-8 <= pv <= pv_hi + 1e-8, f"pv out of range: {pv} not in [{pv_lo}, {pv_hi}]"

        print(f"{sample_id}, family={family}, pv={pv:.6e}, max_slope={max_slope:.4f}, valid_ratio={valid_ratio:.4f}, defect_type={defect_type}")


if __name__ == "__main__":
    main()
