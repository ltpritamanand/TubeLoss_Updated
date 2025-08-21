# Tube Loss for Conformal Prediction

This repository implements **Tube Loss**, a novel loss function for conformal prediction that directly optimizes prediction intervals for both coverage and efficiency.

## Overview

Tube Loss is a piecewise loss function that aims to construct prediction intervals with:
- **High coverage**: The proportion of true values falling within the predicted intervals
- **Narrow width**: Efficient intervals that minimize uncertainty while maintaining coverage

The loss function combines:
1. **Coverage optimization**: Ensures the intervals contain the true values at the desired coverage level
2. **Width regularization**: Encourages narrower intervals for better efficiency
3. **Tube movement**: Adapts the loss based on where predictions fall relative to the target

## Key Features

- 🎯 **Direct optimization** of coverage and efficiency
- 🔧 **Flexible parameters** for different scenarios
- ⚡ **Efficient implementation** using PyTorch Lightning
- 📊 **Built-in metrics** for coverage and interval width
- 🧪 **Example experiments** on real datasets

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- torch
- lightning
- scikit-learn
- pandas
- numpy
- tqdm
- tensorboard

## Quick Start

### Basic Usage

```python
from src.model import TubeLossModel
from src.run_experiment import run_tube_loss_experiment
from data.dataset import GetDataset

# Load your data
X, y = GetDataset("naval", "data/")

# Configure the experiment
config = {
    "dataset_name": "naval",
    "coverage": 0.9,          # Target 90% coverage
    "lr": 0.05,              # Learning rate
    "dropout": 0.2,          # Dropout rate
    "penalty": 0.1,          # Additional penalty term
    "epochs": 400,           # Training epochs
    "batch_size": 1000,      # Batch size
    "test_ratio": 0.2,       # Test split
    "val_ratio": 0.2,        # Validation split
    "random_seed": 42,       # For reproducibility
    "finetuning": False,     # Full training mode
    "device": 0,             # GPU device
    "r": 0.5,               # Tube movement factor
    "delta": 0.03           # Width regularization coefficient
}

# Run the experiment
val_objective, test_results, val_results = run_tube_loss_experiment(config)

print(f"Test Coverage: {test_results['test_coverage']:.4f}")
print(f"Test Width: {test_results['test_width']:.4f}")
```

### Direct Model Usage

```python
import torch
from src.model import TubeLossModel

# Create model
model = TubeLossModel(
    coverage=0.9,      # 90% coverage target
    x_shape=10,        # 10 input features
    hidden_size=64,    # Hidden layer size
    r=0.5,            # Tube movement factor
    delta=0.03        # Width regularization
)

# Forward pass
x = torch.randn(32, 10)  # Batch of 32 samples, 10 features
predictions = model(x)    # Returns [lower_bound, upper_bound]
```

## Tube Loss Parameters

The Tube Loss function has several key parameters:

### Core Parameters
- **`coverage` (q)**: Target coverage level (e.g., 0.9 for 90% intervals)
- **`r`**: Tube movement factor (0.0 to 1.0)
  - Controls how the loss adapts based on where predictions fall
  - r=0.5 provides balanced behavior
- **`delta`**: Width regularization coefficient
  - Controls the trade-off between coverage and interval width
  - Higher values encourage narrower intervals

### Mathematical Formulation

The Tube Loss function is defined as:

```
L = L_coverage + δ × |f₂ - f₁|
```

Where:
- `L_coverage` depends on whether the true value falls within the predicted interval
- `f₁, f₂` are the lower and upper bounds of the prediction interval
- `δ` (delta) is the width regularization coefficient

The coverage component uses different loss terms based on the position of the true value relative to the interval and the tube movement parameter `r`.

## Example Experiment

Run the included naval dataset experiment:

```bash
cd /path/to/TubeLoss_Updated
python run_naval_experiment.py
```

This will:
1. Load the naval propulsion dataset
2. Train a Tube Loss model with optimized hyperparameters
3. Evaluate coverage and interval width on test data
4. Display results

Expected output:
```
--- Naval Dataset Results ---
  Coverage: 0.8911 (89.11%)
  MPIW: 0.0224
-----------------------------
```

## Project Structure

```
TubeLoss_Updated/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_naval_experiment.py      # Example experiment script
├── src/                         # Core implementation
│   ├── __init__.py
│   ├── model.py                # TubeLossModel implementation
│   ├── loss.py                 # Tube Loss function
│   ├── run_experiment.py       # Experiment runner
│   └── metrics.py              # Evaluation metrics
└── data/                       # Dataset utilities
    ├── __init__.py
    ├── dataset.py              # Data loading utilities
    └── datasets/               # Example datasets
        ├── naval.txt
        ├── wine.txt
        └── ...
```

## Key Components

### TubeLossModel (`src/model.py`)
- PyTorch Lightning module implementing the neural network
- Supports flexible architecture with configurable hidden layers
- Includes proper weight initialization and optimization

### Tube_Loss (`src/loss.py`)
- Core implementation of the Tube Loss function
- Handles the piecewise loss computation
- Includes width regularization and tube movement logic

### Experiment Runner (`src/run_experiment.py`)
- Complete experiment pipeline
- Data preprocessing and splitting
- Model training and evaluation
- Results logging and checkpointing

## Hyperparameter Tuning

The key hyperparameters to tune are:

1. **`r` (tube movement factor)**: Start with 0.5, try values between 0.1-0.9
2. **`delta` (width regularization)**: Start with 0.03, try values between 0.01-0.1
3. **`learning_rate`**: Typically 0.001-0.1 depending on dataset
4. **`coverage`**: Set based on your confidence requirements (0.8, 0.9, 0.95)

## Evaluation Metrics

The implementation provides several key metrics:

- **Coverage**: Proportion of true values within predicted intervals
- **MPIW (Mean Prediction Interval Width)**: Average width of prediction intervals
- **Objective**: Combined metric balancing coverage and width

## Citation

If you use this implementation in your research, please cite the original Tube Loss paper:

```bibtex
@article{tubeloss2024,
  title={Tube Loss for Conformal Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
1. Check the existing issues on GitHub
2. Create a new issue with a detailed description
3. Include code examples and error messages when applicable

---

**Note**: This implementation focuses specifically on the Tube Loss function for conformal prediction. It provides a clean, well-documented codebase that can be easily integrated into other projects or extended for specific use cases.
