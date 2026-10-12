"""Шаг 00 — сплиты (CPU).

Для каждого домена SALAD-Bench (base_set, 1-category) — непересекающиеся
harmful_{train,val,test}.json одного размера во всех доменах; выборка стратифицирована по
листу таксономии (3-category). Вопросы, встречающиеся в нескольких доменах или повторно,
выброшены. Harmless — общий для всех доменов срез Alpaca, тоже непересекающийся по сплитам;
он копируется в каталог каждого домена (так их читают DIM-пайплайн и rdo.py).

Пишет results/<model>/splits/<domain>/{harmful,harmless}_{train,val,test}.json
и sha256 каждого файла в manifest.json.
"""
import argparse
import hashlib
import os

import pandas as pd
from datasets import load_dataset

from common import FORK, load_config, read_json, splits_dir, update_manifest, write_json

SPLITS = ("train", "val", "test")


def stratified_sample(pool, n, col, seed):
    """n строк из pool с сохранением долей col (largest remainder: сумма ровно n)."""
    counts = pool[col].value_counts()
    exact = counts / counts.sum() * n
    alloc = exact.astype(int)
    for cat in (exact - alloc).sort_values(ascending=False).index[: n - int(alloc.sum())]:
        alloc[cat] += 1
    parts = [pool[pool[col] == cat].sample(n=int(k), random_state=seed)
             for cat, k in alloc.items() if k > 0]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc = cfg["splits"]
    sizes = {s: sc[f"n_{s}"] for s in SPLITS}
    total = sum(sizes.values())

    # harmless: перемешать один раз, нарезать подряд
    harmless = pd.DataFrame(read_json(os.path.join(FORK, sc["harmless_src"])))
    assert len(harmless) >= total, f"harmless {len(harmless)} < {total}"
    harmless = harmless.sample(frac=1, random_state=sc["seed"]).reset_index(drop=True)
    harmless_splits, i = {}, 0
    for s, n in sizes.items():
        harmless_splits[s] = [{"instruction": q, "source": "alpaca"}
                              for q in harmless["instruction"].iloc[i:i + n]]
        i += n

    df = load_dataset("OpenSafetyLab/Salad-Data", name="base_set")["train"].to_pandas()
    df = df[df["1-category"].isin(cfg["domains"].values())]
    # вопрос, попавший в два домена или повторённый, неоднозначен — выбрасываем все копии
    df = df[~df["question"].duplicated(keep=False)]

    manifest = {"sizes": sizes, "seed": sc["seed"], "harmless_src": sc["harmless_src"],
                "pool": {}, "sha256": {}}
    for d, label in cfg["domains"].items():
        pool = df[df["1-category"] == label]
        assert len(pool) >= total, f"{d}: пул {len(pool)} < {total}"
        sample = stratified_sample(pool, total, "3-category", sc["seed"])
        out = splits_dir(cfg, d)
        i = 0
        for s, n in sizes.items():
            rows = [{"instruction": q, "source": f"salad:{d}", "category": c}
                    for q, c in zip(sample["question"].iloc[i:i + n], sample["3-category"].iloc[i:i + n])]
            i += n
            for harmtype, data in (("harmful", rows), ("harmless", harmless_splits[s])):
                p = os.path.join(out, f"{harmtype}_{s}.json")
                write_json(data, p)
                manifest["sha256"][f"{d}/{harmtype}_{s}"] = sha256(p)
        manifest["pool"][d] = len(pool)
        print(f"{d}: пул {len(pool)} -> {sizes}, листьев {sample['3-category'].nunique()}")

    update_manifest(cfg, "splits", manifest)


if __name__ == "__main__":
    main()
