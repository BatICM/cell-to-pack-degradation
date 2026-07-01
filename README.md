# cell-to-pack-degradation

Data and code to reproduce the analyses in the manuscript **Resolving inter-cell heterogeneity to mitigate degradation amplification in battery packs**.

The repository includes the ESSL1 dataset as a lightweight example. The complete datasets are available at Zenodo: 10.5281/zenodo.20132842.

## I - System requirements

All analyses are conducted in Python. Required packages are listed in `requirements.txt`.

The code has been tested with Python 3.7.12. No non-standard hardware is required for the ESSL1 example. GPU acceleration can be used for model training but is not required for setup checking or pack-level evaluation.

## II - Installation

Create and activate a Python environment:

```bash
conda create -n ctp_deg python=3.7
conda activate ctp_deg
pip install -r requirements.txt
```
Typical installation time: ~5–10 min on a normal desktop computer

## III - Demos and instructions

Check the local setup:

```bash
python scripts/check_setup.py --config configs/essl1.yaml
```

Train the cell-level model:

```bash
python scripts/train_cell_model.py --config configs/essl1.yaml
```

Infer pack states:

```bash
python scripts/infer_pack_states.py --config configs/essl1.yaml
```

Estimate pack capacity:

```bash
python scripts/estimate_pack_capacity.py --config configs/essl1.yaml
```

Generated files are saved under `outputs/`.

## IV - Data

The example data are stored in:

```text
data/raw/ESSL1/
```

For full-data reproduction, place the complete datasets under:

```text
data/raw/
├── ESSL1/
├── ESSL2/
├── PBSL1/
└── ESSH1/
```

Large datasets, trained checkpoints, and generated outputs are not tracked in GitHub.

## V - Troubleshooting

If `check_setup.py` reports missing files, check whether the data paths in `configs/*.yaml` match the local file structure.

If CUDA is unavailable, run the scripts on CPU or reduce the batch size in the configuration file.

## Citation

Citation information will be added after publication.
