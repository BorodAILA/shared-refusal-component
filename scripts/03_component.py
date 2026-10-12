"""Шаг 03 — общая компонента всех 24 векторов (CPU).

Популяция: 4 DIM + 20 одномерных конусов RDO, все единичные, веса равные:
    P = (1/N) sum_i v_i v_i^T,   N = 24.
Общая компонента w — первый собственный вектор P; lam_1 = mean_i cos^2(w, v_i).
Знак w выбран так, чтобы cos(w, средний DIM) > 0 (добавление w наводит отказ).

Спектр берётся через малую матрицу Грама G = V V^T / N (24 x 24): ненулевые с.з. P и G
совпадают, w = V^T u_1 / ||V^T u_1||.

Нулевой уровень: N независимых случайных единичных векторов той же размерности. Нижняя
граница lam_1 = 1/N (все векторы взаимно ортогональны), потолок 1 (все совпадают).

Пишет results/<model>/component/{component.pt, component.json} и reports/component.md.
"""
import argparse
import os

import torch

from common import load_config, load_vectors, results_dir, update_manifest, write_json, write_report


def spectrum(V):
    N = V.shape[0]
    lam, U = torch.linalg.eigh(V @ V.T / N)
    order = torch.argsort(lam, descending=True)
    lam, U = lam[order], U[:, order]
    W = V.T @ U
    return lam, W / W.norm(dim=0, keepdim=True).clamp(min=1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["component"]

    names, V = load_vectors(cfg)
    N, hidden = V.shape
    lam, W = spectrum(V)
    w = W[:, 0]
    mean_dim = V[[i for i, n in enumerate(names) if n.startswith("dim_")]].mean(0)
    if w @ mean_dim < 0:
        w = -w
    cos = V @ w                                            # [N], со знаком
    assert torch.allclose((cos ** 2).mean(), lam[0], atol=1e-8)

    g = torch.Generator().manual_seed(cc["null_seed"])
    null = []
    for _ in range(cc["n_null"]):
        R = torch.randn(N, hidden, generator=g, dtype=torch.float64)
        null.append(spectrum(R / R.norm(dim=1, keepdim=True))[0][0])
    null = torch.stack(null)
    null_mean, null_p95, null_max = float(null.mean()), float(null.quantile(0.95)), float(null.max())

    out = results_dir(cfg, "component")
    os.makedirs(out, exist_ok=True)
    torch.save({"w": w.float(), "lam": lam[:8].float(), "names": names, "cos": cos.float()},
               f"{out}/component.pt")
    summary = {"N": N, "hidden": hidden, "lam": [float(x) for x in lam[:8]],
               "null": {"mean": null_mean, "p95": null_p95, "max": null_max, "n": cc["n_null"]},
               "cos": dict(zip(names, map(float, cos)))}
    write_json(summary, f"{out}/component.json")
    update_manifest(cfg, "component", {"lam1": float(lam[0]), "null_p95": null_p95})

    lines = [f"# Общая компонента — {cfg['model_alias']}", "",
             f"N = {N} векторов (4 DIM + 20 RDO), равные веса, hidden = {hidden}.", "",
             "| | λ1 | λ2 | λ3 | нуль: среднее | нуль: p95 | нижняя граница 1/N |",
             "|---|---|---|---|---|---|---|",
             f"| | **{lam[0]:.3f}** | {lam[1]:.3f} | {lam[2]:.3f} | {null_mean:.3f} | {null_p95:.3f} | {1 / N:.3f} |",
             "", "cos(w, v) по векторам (знак w — по среднему DIM):", "",
             "| вектор | cos | cos² |", "|---|---|---|"]
    lines += [f"| {n} | {c:+.3f} | {c * c:.3f} |" for n, c in zip(names, cos.tolist())]
    write_report(cfg, "component.md", lines)
    print(f"lam1={lam[0]:.3f}  lam2={lam[1]:.3f}  нуль p95={null_p95:.3f}")


if __name__ == "__main__":
    main()
