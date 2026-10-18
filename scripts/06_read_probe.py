"""Шаг 06 — read против write по слоям (GPU, только forward, ничего не обучается).

Перенос §8 common_axis (concept_dim, Ф. Захаров): детектирует ли остаток DIM вредность, будучи
причинно пустым, и на каком слое «включается» общая компонента. Для каждого слоя берётся
активация последнего токена на ВХОДЕ слоя (resid_pre — как при выборе точки DIM в шаге 01),
проецируется на фиксированное направление, и по этим скалярам считается AUROC harmful/harmless.

Выборка: harmful — по n_per_domain из каждого домена (train+val+test, дедуп), harmless — весь
общий пул (train+val+test). AUROC ранговый и порога не требует, поэтому считается на всём пуле;
95% ДИ — бутстрэп на фиксированных слоях (0, середина, слой DIM, последний); точность —
k-фолдовая кросс-валидация с подбором порога.

Направления (единичные и фиксированные; меняется только слой, где читается активация):
  dim            — DIM объединённой выборки (dim/union) в точке шага 01;
  w              — общая компонента только по 20 RDO (component_rdo: DIM в построении нет);
  dim_perp_w     — остаток DIM без w (кандидат в read);
  w_perp_dim     — w без DIM-части (кандидат в write);
  rand           — случайное;
  meandiff_layer — разность средних на этом же слое по тому же пулу: потолок слоя;
  act_norm       — детектор по самой ‖h‖: мера конфаундера масштаба.

Нужны готовые шаги 00–02 и шаг 03 с COMPONENT=rdo.
Пишет results/<model>/read_probe/read_probe.json и reports/read_probe.md.
"""
import argparse
import json
import os
import random

import torch

from common import (TEMPLATES, dim_dir, family, load_config, read_json, read_manifest, results_dir,
                    splits_dir, unit, write_json, write_report)


# --- статистика: как в concept_dim/experiments/common_axis/scripts/read_probe.py ---

def auroc(pos, neg):
    """Ранговый AUROC; знак направления произволен -> max(auc, 1-auc)."""
    x = torch.cat([pos, neg])
    r = torch.empty_like(x)
    r[x.argsort()] = torch.arange(len(x), dtype=x.dtype)
    a = ((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))).item()
    return max(a, 1 - a)


def best_threshold(pos, neg):
    """Порог максимальной точности векторно через cumsum, в обеих ориентациях."""
    v = torch.cat([pos, neg])
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    order = v.argsort()
    vs, ys = v[order], y[order]
    cp, cn = torch.cumsum(ys, 0), torch.cumsum(1 - ys, 0)
    P, N = len(pos), len(neg)
    acc_pos = ((P - cp) + cn) / (P + N)
    acc_neg = (cp + (N - cn)) / (P + N)
    i_p, i_n = int(acc_pos.argmax()), int(acc_neg.argmax())
    mid = lambda i: float(vs[i] if i == len(vs) - 1 else (vs[i] + vs[i + 1]) / 2)
    if acc_pos[i_p] >= acc_neg[i_n]:
        return mid(i_p), 1, float(acc_pos[i_p])
    return mid(i_n), -1, float(acc_neg[i_n])


def cv_accuracy(pos, neg, k, seed):
    """Точность с порогом: k-фолд, порог на k-1 фолдах, замер на отложенном."""
    g = torch.Generator().manual_seed(seed)
    ip, in_ = torch.randperm(len(pos), generator=g), torch.randperm(len(neg), generator=g)
    accs = []
    for f in range(k):
        hp, hn = ip[f::k], in_[f::k]
        mp = torch.ones(len(pos), dtype=torch.bool); mp[hp] = False
        mn = torch.ones(len(neg), dtype=torch.bool); mn[hn] = False
        t, sgn, _ = best_threshold(pos[mp], neg[mn])
        tp, tn = pos[hp], neg[hn]
        accs.append(float((((tp - t) * sgn > 0).sum() + ((tn - t) * sgn <= 0).sum()) / (len(tp) + len(tn))))
    return sum(accs) / len(accs)


def auroc_ci(pos, neg, n_boot, seed):
    """95% ДИ AUROC бутстрэпом по промптам, стратифицированно по классам."""
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_boot):
        p = pos[torch.randint(len(pos), (len(pos),), generator=g)]
        n = neg[torch.randint(len(neg), (len(neg),), generator=g)]
        vals.append(auroc(p, n))
    v = torch.tensor(vals).sort().values
    return float(v[int(0.025 * n_boot)]), float(v[int(0.975 * n_boot) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rp = {"n_per_domain": 300, "batch_size": 16, "folds": 5, "n_boot": 1000, "seed": 21, **cfg.get("read_probe", {})}
    domains = list(cfg["domains"])
    torch.manual_seed(rp["seed"])

    # --- выборка ---
    template = TEMPLATES[family(cfg["model"])]
    rnd = random.Random(rp["seed"])
    seen, pos_txt = set(), []
    for d in domains:
        pool = []
        for s in ("train", "val", "test"):
            for r in read_json(os.path.join(splits_dir(cfg, d), f"harmful_{s}.json")):
                if r["instruction"] not in seen:
                    seen.add(r["instruction"])
                    pool.append(r["instruction"])
        pos_txt += rnd.sample(pool, min(rp["n_per_domain"], len(pool)))
    neg_txt = list(dict.fromkeys(r["instruction"] for s in ("train", "val", "test")
                                 for r in read_json(os.path.join(splits_dir(cfg, domains[0]), f"harmless_{s}.json"))))
    print(f"пул: {len(pos_txt)} harmful ({rp['n_per_domain']} на домен) + {len(neg_txt)} harmless", flush=True)

    # --- направления ---
    d = unit(torch.load(os.path.join(dim_dir(cfg, "union"), "direction.pt"), map_location="cpu", weights_only=True).float())
    w = unit(torch.load(results_dir(cfg, "component_rdo", "component.pt"), map_location="cpu", weights_only=True)["w"].float())
    g = torch.Generator().manual_seed(rp["seed"])
    dirs = {"dim": d, "w": w,
            "dim_perp_w": unit(d - (d @ w) * w),
            "w_perp_dim": unit(w - (w @ d) * d),
            "rand": unit(torch.randn(d.shape[0], generator=g))}
    dim_layer = int(read_manifest(cfg)["dim"]["layer"])

    # --- активации последнего токена на входе каждого слоя ---
    from nnsight import LanguageModel
    model = LanguageModel(cfg["model"], cache_dir=os.getenv("HUGGINGFACE_CACHE_DIR"),
                          device_map="auto", torch_dtype=torch.bfloat16)
    model.requires_grad_(False)
    with model.trace("Hello"):   # загрузить веса на GPU
        pass

    def acts(texts):
        prompts, bs, out = [template.format(instruction=t) for t in texts], rp["batch_size"], []
        for i in range(0, len(prompts), bs):
            with model.trace(prompts[i:i + bs]):
                per = [layer.input[:, -1].save() for layer in model.model.layers]
            out.append(torch.stack([p.value.detach().half().cpu() for p in per], dim=1))   # [b, L, hidden]
            torch.cuda.empty_cache()
        return torch.cat(out, 0)

    A_pos, A_neg = acts(pos_txt), acts(neg_txt)
    n_layers = A_pos.shape[1]
    print(f"активации сняты: {n_layers} слоёв, точка DIM — слой {dim_layer}", flush=True)

    fixed = sorted({0, n_layers // 2, dim_layer, n_layers - 1})
    names = list(dirs) + ["meandiff_layer", "act_norm"]
    rows = []
    for L in range(n_layers):
        HP, HN = A_pos[:, L].float(), A_neg[:, L].float()
        md = unit(HP.mean(0) - HN.mean(0))
        for name in names:
            if name == "act_norm":
                sp, sn = HP.norm(dim=1), HN.norm(dim=1)
            else:
                v = md if name == "meandiff_layer" else dirs[name]
                sp, sn = HP @ v, HN @ v
            pooled = ((sp.var() + sn.var()) / 2).sqrt()
            r = {"layer": L, "direction": name, "auroc": auroc(sp, sn),
                 "acc_cv": cv_accuracy(sp, sn, rp["folds"], rp["seed"]),
                 "cohen_d": float((sp.mean() - sn.mean()).abs() / pooled.clamp(min=1e-9))}
            if L in fixed:
                r["auroc_lo"], r["auroc_hi"] = auroc_ci(sp, sn, rp["n_boot"], rp["seed"])
            rows.append(r)

    cos = {"dim_w": float(d @ w), "dim_perp_w_dim": float(dirs["dim_perp_w"] @ d),
           "w_perp_dim_w": float(dirs["w_perp_dim"] @ w)}
    write_json({"n_pos": len(pos_txt), "n_neg": len(neg_txt), "n_layers": n_layers, "dim_layer": dim_layer,
                "fixed_layers": fixed, "cos": cos, "params": rp, "rows": rows},
               results_dir(cfg, "read_probe", "read_probe.json"))

    cell = lambda l, n, k: next(r[k] for r in rows if r["layer"] == l and r["direction"] == n)
    lines = [f"# Read против write по слоям — {cfg['model_alias']}", "",
             f"Пул: **{len(pos_txt)} harmful** ({rp['n_per_domain']} на домен, train+val+test) + "
             f"**{len(neg_txt)} harmless**. Активация последнего токена на входе слоя (resid_pre). "
             f"AUROC = max(auc, 1−auc) на всём пуле, 95% ДИ — бутстрэп ({rp['n_boot']}); "
             f"w — только по 20 RDO. cos(DIM, w) = {cos['dim_w']:.3f}.", "",
             f"## Фиксированные слои (точка DIM — слой {dim_layer})", "",
             "| направление | " + " | ".join(f"слой {l}" for l in fixed) + " |", "|" + "---|" * (len(fixed) + 1)]
    lines += [f"| {n} | " + " | ".join(f"{cell(l, n, 'auroc'):.3f} <sub>[{cell(l, n, 'auroc_lo'):.3f}, "
                                        f"{cell(l, n, 'auroc_hi'):.3f}]</sub>" for l in fixed) + " |" for n in names]
    lines += ["", "## По слоям (AUROC)", "", "| слой | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    lines += [f"| {l} | " + " | ".join(f"{cell(l, n, 'auroc'):.3f}" for n in names) + " |" for l in range(n_layers)]
    write_report(cfg, "read_probe.md", lines)
    for n in names:
        print(f"{n:15s} " + "  ".join(f"L{l}={cell(l, n, 'auroc'):.3f}" for l in fixed), flush=True)


if __name__ == "__main__":
    main()
