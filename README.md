# cell-to-pack-degradation

Data and code for the manuscript **Resolving inter-cell heterogeneity to mitigate degradation amplification in battery packs**.

This repository provides the source code, configuration files, and a lightweight ESSL1 example dataset for running the main cell-to-pack degradation workflow. The complete datasets are available at Zenodo: [https://doi.org/10.5281/zenodo.20132842](https://doi.org/10.5281/zenodo.20132842)

A model card is provided in [MODEL_CARD.md](MODEL_CARD.md). It describes the intended use, model architecture, inputs and outputs, training data, data splitting strategy, evaluation metrics, limitations, out-of-scope uses, and ethical considerations for the proposed cell-to-pack degradation analysis model.

## I - System requirements

All analyses are conducted in Python. Required packages are listed in `requirements.txt`.

Tested environment:

- OS: Windows 11
- CPU: 13th Gen Intel(R) Core(TM) i7-13700F
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 3070, 8 GB
- Python: 3.7.12
- PyTorch: 1.13.1+cu117
- CUDA: 11.7

No non-standard hardware is required for setup checking and pack-level evaluation. A CUDA-enabled GPU is optional for model training and was used for the runtime reported below.

## II - Installation

Clone the repository:

```bash
git clone https://github.com/BatICM/cell-to-pack-degradation.git
cd cell-to-pack-degradation
```

Create and activate a Python environment:

```bash
conda create -n ctp_deg python=3.7
conda activate ctp_deg
pip install -r requirements.txt
```

Typical installation time: approximately 5–10 min on a normal desktop computer with a stable internet connection.

## III - Repository structure

```text
configs/              Configuration files for datasets and model settings
data/                 Example data and local dataset directory
scripts/              Main scripts for setup checking, training, inference and capacity estimation
src/                  Source code for data processing, models, training and evaluation
outputs/              Generated checkpoints, predictions, tables and figures
requirements.txt      Python dependencies
LICENSE               Software license
README.md             Instructions for installation, demo and use
MODEL_CARD.md         Model card describing intended use, architecture, data, evaluation and limitations
```

## IV - Demo

The ESSL1 dataset included in this repository is provided as a lightweight example. The demo workflow includes setup checking, cell-level model training, pack-state inference, and pack-capacity estimation.

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

## V - Expected output

After running the ESSL1 demo workflow, generated files are saved under `outputs/`.

The cell-level training script generates:

```text
outputs/checkpoints/best_dual_cnn_lstm_model_essl1.pth
outputs/predictions/essl1_training_summary.pkl
outputs/predictions/essl1_cell_predictions.npz
outputs/figures/essl1_training_loss.png
outputs/figures/essl1_train_test_results.png
```

If runtime logging is enabled, it also generates:

```text
outputs/predictions/essl1_runtime_summary.json
outputs/predictions/essl1_runtime_summary.txt
```

The pack-state inference script generates the pack prediction file specified by `pack_prediction_file` in `configs/essl1.yaml`. This file contains predicted SOH/SOC states, real pack SOH labels, sample indices, and pack channel information.

The pack-capacity estimation script generates:

```text
outputs/predictions/essl1_pack_metrics.json
outputs/figures/essl1_pack_estimation.png
```

It also saves the pack capacity estimation table to the path specified by `pack_capacity_table` in `configs/essl1.yaml`.

## VI - Runtime

The runtime below was measured using the lightweight ESSL1 example dataset included in this repository.

- Cell-level model training and evaluation: approximately 5.2 min
- Other demo steps, including setup checking, pack-state inference and pack-capacity estimation: <1 min each

Runtime was measured on Windows 11 with an Intel Core i7-13700F CPU, 32 GB RAM, and an NVIDIA GeForce RTX 3070 GPU with 8 GB memory.

Runtime may vary depending on hardware, software environment, dataset size, and configuration.

## VII - Data

The example data are stored in:

```text
data/raw/ESSL1/
```

For full-data reproduction, download the complete datasets from Zenodo and place them under:

```text
data/raw/
├── ESSL1/
├── ESSL2/
├── PBSL1/
└── ESSH1/
```

Large datasets, trained checkpoints, and generated outputs are not tracked in GitHub.

## VIII - Running on custom data

To run the code on custom data, organize the input files following the same structure as the ESSL1 example dataset. Then update the corresponding configuration file in `configs/` and run the scripts with the new config file.

Example:

```bash
python scripts/train_cell_model.py --config configs/your_dataset.yaml
python scripts/infer_pack_states.py --config configs/your_dataset.yaml
python scripts/estimate_pack_capacity.py --config configs/your_dataset.yaml
```

The required input format can be checked from the ESSL1 example data and `configs/essl1.yaml`.

## IX - Notes

The ESSL1 dataset is provided as a lightweight example for checking the code workflow. Full-data reproduction requires downloading the complete datasets from Zenodo.

Do not change the directory structure of the downloaded datasets unless the paths in the corresponding configuration files are also updated.

## X - Troubleshooting

If `check_setup.py` reports missing files, check whether the data paths in `configs/*.yaml` match the local file structure.

If CUDA is unavailable, run the scripts on CPU or reduce the batch size in the configuration file.

## XI - License

This project is released under the MIT License. See the `LICENSE` file for details.

## XII - Citation

Citation information will be added after publication.

