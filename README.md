# cell-to-pack-degradation

Data and code necessary to replicate the analyses in the manuscript  
**"State-space geometry drives degradation amplification in battery packs"**.

The repository includes the ESSL1 dataset as a lightweight example. The complete datasets used in the manuscript will be deposited in Zenodo and linked here after finalization.

## I - System requirements

The analyses are conducted in Python. All necessary Python packages are listed in `requirements.txt`.

The code has been tested with Python 3.8. No non-standard hardware is required for running the ESSL1 example dataset. GPU acceleration can be used for model training if available, but is not required for checking the data structure or running the main evaluation scripts.

The repository contains:

```text
configs/        Dataset-specific configuration files
scripts/        Executable scripts for setup checking, training, inference, and evaluation
src/            Core source code
data/raw/ESSL1  Example dataset
requirements.txt
```

Large datasets, trained checkpoints, and generated outputs are not tracked in GitHub.

## II - Installation

Python can be downloaded from:

https://www.python.org/downloads/

A virtual environment can be created using conda or venv. For example, using conda:

```bash
conda create -n ctp_deg python=3.8
conda activate ctp_deg
pip install -r requirements.txt
```

Alternatively, using venv:

```bash
python -m venv ctp_deg
```

On Windows:

```bash
ctp_deg\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
source ctp_deg/bin/activate
pip install -r requirements.txt
```

## III - Demos and Instructions

The ESSL1 example dataset is included in this repository and can be used to check the expected data structure and run the main workflow.

First, check the local setup:

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

Generated files are saved under:

```text
outputs/
├── checkpoints/
├── predictions/
└── tables/
```

The `outputs/` directory is ignored by Git.

To run the other datasets, download the complete data from Zenodo and place the files under:

```text
data/raw/
├── ESSL1/
├── ESSL2/
├── PBSL1/
└── ESSH1/
```

Then replace the configuration file, for example:

```bash
python scripts/check_setup.py --config configs/essl2.yaml
python scripts/train_cell_model.py --config configs/essl2.yaml
python scripts/infer_pack_states.py --config configs/essl2.yaml
python scripts/estimate_pack_capacity.py --config configs/essl2.yaml
```

## IV - Troubleshooting

If `check_setup.py` reports missing files, check whether the dataset is placed under the expected path defined in the corresponding YAML file in `configs/`.

If CUDA is unavailable or GPU memory is insufficient, use CPU execution or reduce the batch size in the configuration file.

If package import errors occur, reinstall the dependencies with:

```bash
pip install -r requirements.txt
```

If path-related errors occur on Windows, avoid moving the repository after installation and make sure the command line is opened from the repository root directory.

## Citation

Citation information will be added after publication.

## Data availability

The complete raw and processed datasets will be deposited in Zenodo. The Zenodo record will be added here after finalization.

## License

License information will be added before public release.
