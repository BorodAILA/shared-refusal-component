# shared-refusal-component

Code to reproduce the experiments of the paper (TBD).

We take four harm domains of SALAD-Bench — O1 Representation & Toxicity, O2 Misinformation,
O5 Malicious Use, O6 Human Autonomy — and for each of them compute a difference-in-means
(DIM) refusal direction and train five one-dimensional refusal cones (RDO, Wollschläger et
al.) with distinct seeds. All 24 vectors are intersected with equal weights to obtain one
shared component; we then measure its share of each DIM and test it with directional
ablation.

## Pipeline

| step | resource | what it does |
|---|---|---|
| `00_splits` | CPU | train/val/test splits for each domain + shared harmless set |
| `01_dim` | GPU | DIM per domain; one extraction point (pos, layer) for all domains |
| `02_rdo` | GPU ×20 | one-dimensional RDO cones, seeds 1–20 (5 per domain) |
| `03_component` | CPU | shared component of all 24 vectors, null band |
| `04_energy` | CPU | share of the component in each DIM |
| `05_ablate` | GPU | ASR under ablation at matched harmless KL |

Configuration: [`configs/qwen2.5-3b.yaml`](configs/qwen2.5-3b.yaml).
All artifacts go to `results/<model>/`.
