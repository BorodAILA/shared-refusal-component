"""Общие пути, конфиг и manifest для шагов 00–05."""
import json
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK = os.path.abspath(os.getenv("FORK", os.path.join(REPO, "..", "geometry-of-refusal")))


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["model_alias"] = cfg["model"].split("/")[-1]
    return cfg


def results_dir(cfg, *parts):
    return os.path.join(REPO, "results", cfg["model_alias"], *parts)


def splits_dir(cfg, domain):
    return results_dir(cfg, "splits", domain)


def dim_dir(cfg, domain):
    return results_dir(cfg, "dim", domain)


def rdo_dir(cfg, domain):
    return results_dir(cfg, "rdo", domain)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def update_manifest(cfg, key, value):
    """manifest.json — что получено каждым шагом (точка извлечения, сиды, хэши сплитов)."""
    path = results_dir(cfg, "manifest.json")
    m = read_json(path) if os.path.exists(path) else {"model": cfg["model"]}
    m[key] = value
    write_json(m, path)


def read_manifest(cfg):
    return read_json(results_dir(cfg, "manifest.json"))


def write_report(cfg, name, lines):
    path = results_dir(cfg, "reports", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[отчёт: {path}]")


def unit(v):
    return v / v.norm()


# Шаблоны чата и refusal-токены для шагов 05–06 — те же, что в rdo.py (иначе метрика не та, что при обучении)
TEMPLATES = {
    "qwen2.5": ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                "<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"),
    "llama-3": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
}
REFUSAL_TOKS = {"qwen2.5": [40, 2121], "llama-3": [40]}   # 'I', 'As' / 'I' — как в rdo.py


def family(model):
    for k in TEMPLATES:
        if k in model.lower():
            return k
    raise ValueError(f"нет шаблона и refusal-токенов для {model}")


# Из чего строится общая компонента (шаги 03–05): all — 4 DIM + 20 RDO; rdo — только 20 RDO,
# тогда сравнение с DIM не круговое. Вариант rdo пишет в component_rdo/, energy_rdo/, ablation_rdo/.
COMPONENT = os.getenv("COMPONENT", "all")
assert COMPONENT in ("all", "rdo"), f"COMPONENT={COMPONENT}: ожидается all или rdo"


def variant(name):
    """component → component_rdo, ablation.md → ablation_rdo.md; при COMPONENT=all имя не меняется."""
    if COMPONENT == "all":
        return name
    stem, dot, ext = name.partition(".")
    return f"{stem}_{COMPONENT}{dot}{ext}"


def load_vectors(cfg):
    """Все 24 вектора в фиксированном порядке: по доменам — DIM, затем RDO по сидам.
    Возвращает (names, V [N, hidden] float64, единичные строки)."""
    import torch

    names, vecs = [], []
    for d in cfg["domains"]:
        vecs.append(torch.load(os.path.join(dim_dir(cfg, d), "direction.pt"), map_location="cpu", weights_only=True))
        names.append(f"dim_{d}")
        for s in cfg["rdo"]["seeds"][d]:
            p = os.path.join(rdo_dir(cfg, d), f"seed_{s:02d}.pt")
            vecs.append(torch.load(p, map_location="cpu", weights_only=True)["vector"])
            names.append(f"rdo_{d}_s{s:02d}")
    V = torch.stack([unit(v.double()) for v in vecs])
    return names, V
