# Model Card

## Model name

Physics-regularized dual CNN-LSTM model for joint SOH/SOC estimation.

## Model description

This model is used for joint estimation of cell-level state of health (SOH) and state of charge (SOC) from lithium-ion battery cycling data. The estimated cell states are further used for reconstructing pack-level cell-state distributions and estimating pack-level capacity degradation.

The model supports the cell-to-pack degradation analysis described in the manuscript:

**Resolving inter-cell heterogeneity to mitigate degradation amplification in battery packs**

## Intended use

This model is intended for research use in:

- lithium-ion battery degradation analysis;
- cell-level SOH and SOC estimation;
- reconstruction of cell-state distributions in battery packs;
- analysis of inter-cell heterogeneity and degradation amplification;
- pack-level capacity estimation using inferred cell states.

The model is designed for offline analysis and research reproduction. It is not intended for direct deployment as a safety-critical battery management system.

## Model architecture

The model uses a dual-branch CNN-LSTM-FC architecture.

### SOH estimator

The SOH branch estimates cell SOH from intermediate-timescale charging samples.

Architecture:

- 1D convolution layer: input channels → 16 channels
- Batch normalization + ReLU
- 1D convolution layer: 16 → 32 channels
- Batch normalization + ReLU
- 1D convolution layer: 32 → 30 channels
- Batch normalization + ReLU
- Optional dropout
- One-layer LSTM
- Fully connected regression head
- ReLU output activation

The default hidden dimension of the LSTM is 32.

### SOC estimator

The SOC branch estimates cell SOC from short-timescale charging samples.

The SOC branch uses the predicted SOH as an additional conditioning feature. The cycle-level SOH estimate is averaged and concatenated with the SOC input sequence before feature extraction.

Architecture:

- Input SOC features + predicted SOH conditioning feature
- 1D convolution layer: input channels + 1 → 16 channels
- Batch normalization + ReLU
- 1D convolution layer: 16 → 32 channels
- Batch normalization + ReLU
- 1D convolution layer: 32 → 30 channels
- Batch normalization + ReLU
- Optional dropout
- One-layer LSTM
- Fully connected regression head
- ReLU output activation
- Output clipped to the range [0, 1]

The default hidden dimension of the LSTM is 32.

### Prediction order

The model first predicts SOH using the SOH branch. The SOC branch then uses the predicted SOH as a conditioning input to estimate SOC.

## Inputs

The model uses multi-timescale battery cycling samples.

Typical input information includes:

- voltage;
- current;
- temperature;
- charge capacity;
- cycle information;
- operating-condition variables such as current rate and temperature.

For the SOH branch, intermediate-timescale samples are used.  
For the SOC branch, short-timescale samples are used.

The required input format can be checked from the ESSL1 example data and `configs/essl1.yaml`.

## Outputs

The model outputs:

- cell-level SOH estimates;
- cell-level SOC estimates.

In the full workflow, these outputs are further used to generate:

- reconstructed pack-level cell-state distributions;
- pack-level capacity estimation results;
- evaluation metrics;
- figures and result tables.

Generated files are saved under the `outputs/` directory.

## Training data

The model is trained using in-house cell-level lithium-ion battery aging data.

The repository includes the ESSL1 dataset as a lightweight example. The complete datasets are available at Zenodo:

```text
[https://doi.org/10.5281/zenodo.20132842](https://doi.org/10.5281/zenodo.21105057)
```

The full dataset includes multiple cell-pack pairs, chemistries, capacities, pack configurations, and operating scenarios.

## Data splitting

Cell-level data are used for model training and model selection. The data are split into training and validation/testing subsets according to the configuration file.

Pack-level data are not used for model training or model selection. They are used for independent pack-level evaluation after cell-level model training.

This design reflects the intended cell-to-pack application: the model learns from cell-level aging data and is then evaluated on pack-level data.

## Training objective

The model is trained with a multi-objective loss for joint SOH/SOC estimation.

The training objective includes:

- SOH estimation loss;
- SOC estimation loss;
- charge-conservation consistency;
- degradation-consistency regularization.

These physics-related constraints are used to encourage physically consistent state estimation across different timescales.

## Evaluation

Model performance is evaluated using error metrics for SOH, SOC, and pack-level capacity estimation.

Main metrics include:

- mean absolute error (MAE);
- root mean square error (RMSE);
- mean absolute percentage error (MAPE), where applicable.

The pack-level evaluation compares the proposed method with the conventional weakest-cell baseline. Ablation experiments are used to evaluate the contribution of physics-based constraints.

## Demo workflow

The ESSL1 demo workflow can be run using:

```bash
python scripts/check_setup.py --config configs/essl1.yaml
python scripts/train_cell_model.py --config configs/essl1.yaml
python scripts/infer_pack_states.py --config configs/essl1.yaml
python scripts/estimate_pack_capacity.py --config configs/essl1.yaml
```

The demo includes:

1. setup checking;
2. cell-level model training;
3. pack-state inference;
4. pack-capacity estimation.

## Expected output

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

The pack-state inference script generates the pack prediction file specified by `pack_prediction_file` in `configs/essl1.yaml`.

The pack-capacity estimation script generates:

```text
outputs/predictions/essl1_pack_metrics.json
outputs/figures/essl1_pack_estimation.png
```

It also saves the pack capacity estimation table to the path specified by `pack_capacity_table` in `configs/essl1.yaml`.

## Runtime

The ESSL1 demo was tested on the following environment:

- OS: Windows 11
- CPU: 13th Gen Intel(R) Core(TM) i7-13700F
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 3070, 8 GB
- Python: 3.7.12
- PyTorch: 1.13.1+cu117
- CUDA: 11.7

Typical runtime for the ESSL1 example:

- Cell-level model training and evaluation: approximately 5.2 min
- Setup checking, pack-state inference, and pack-capacity estimation: less than 1 min each

Runtime may vary depending on hardware, software environment, dataset size, and configuration.

## Limitations

The model was developed and evaluated using the datasets described in the associated manuscript and repository.

Potential limitations include:

- generalization to unseen battery chemistries, formats, capacities, or pack configurations may require additional validation;
- performance may depend on the quality and consistency of voltage, current, temperature, and capacity measurements;
- pack-level inference assumes that input files follow the required data format and preprocessing pipeline;
- field data may contain noise, missing segments, and operating-condition variability;
- the model is intended for offline research analysis rather than real-time control.

## Out-of-scope use

This model should not be directly used for:

- safety-critical battery management;
- real-time pack control;
- fault diagnosis or safety protection;
- warranty decisions;
- deployment on unseen battery systems without additional validation.

## Ethical and safety considerations

The model is intended to support research on battery degradation and resource utilization. Incorrect use of the model for operational battery control without additional validation may lead to unreliable decisions. Any practical deployment should include independent validation, safety checks, and domain-specific engineering review.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Citation

Citation information will be added after publication.
