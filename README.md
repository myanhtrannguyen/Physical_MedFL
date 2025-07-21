# Federated Learning Project for Medical Image Segmentation

This project provides a research-grade framework for conducting federated learning research on medical image segmentation tasks using the ACDC dataset. Built with **Flower 1.18.0+**, it implements **UnifiedFairnessStrategy** with advanced metrics tracking, experiment management, and publication-ready features.

## Key Features

- **Modern Flower App Architecture**: Uses latest `flwr run` CLI and app structure
- **Research-Grade Implementation**: Q1/Q2 publication standard with comprehensive metrics and statistical analysis  
- **Flower 1.18.0+ Compatible**: Latest federated learning framework with UnifiedFairnessStrategy
- **Advanced Metrics**: Dice score, IoU, sensitivity, specificity, HD95, ASSD tracking
- **Experiment Management**: Comprehensive logging, configuration management, and reproducibility
- **Client Heterogeneity**: Support for various federated learning scenarios
- **Statistical Analysis**: Built-in statistical tests and convergence monitoring
- **Export Capabilities**: CSV, JSON, and checkpoint export for further analysis

## Project Structure

```
Federated Learning/
├── 📄 README.md                # This documentation
├── 📄 pyproject.toml          # Legacy project configuration
├── 🔧 env.sh                  # Environment setup and activation script
├── 📄 LICENSE                 # Apache 2.0 License
│
├── med/                    # 🌟 Modern Flower App
│   ├── pyproject.toml        # App configuration & dependencies
│   └── med/                  # Core app modules
│       ├── client_app.py     # Flower client app
│       ├── server_app.py     # Flower server app  
│       ├── task.py          # Training/evaluation logic
│       ├── utils.py         # Post-processing utilities
│       └── __init__.py      # Package exports
│
├── src/                    # Legacy source code (used by Flower app)
│   ├── federated/            # Main federated learning components
│   │   ├── client.py         # Research-grade Flower client
│   │   └── server.py         # Research-grade Flower server
│   ├── models/               # Model architectures
│   │   └── unet.py          # U-Net for medical segmentation
│   ├── utils/                # Utility functions
│   │   ├── losses.py        # Loss functions (Adaptive tvMF Dice)
│   │   └── metrics.py       # Evaluation metrics
│   └── data_handling/        # Data loading and preprocessing
│       └── data_loader.py   # ACDC dataset loader
│
├── data/                  # Dataset storage
│   └── ACDC_preprocessed/   # Preprocessed ACDC dataset
│       ├── ACDC_training_volumes/
│       ├── ACDC_training_slices/
│       └── ACDC_testing_volumes/
│
├── research_exports/      # Experiment results & exports
│   └── [experiment_name]/ # Individual experiment outputs
│
├── logs/                  # Experiment logs
│   └── clients/             # Client-specific logs
│
├── checkpoints/           # Model checkpoints
├── server_checkpoints/    # Server-side checkpoints
└── venv/                  # Virtual environment
```

## Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to project
cd "Federated Learning"

# Setup environment (first time)
./env.sh setup

# Activate environment (daily use)
source env.sh
```

### 2. Run Federated Learning Experiment

```bash
# Run with default configuration
flwr run med

# Or run from med directory
cd med
flwr run .
```

### 3. View Results

After training completes, results are automatically exported to:
- `research_exports/[experiment_name]/`
- Includes CSV files, plots, and experiment summary

## Configuration

Edit `med/pyproject.toml` to modify experiment parameters:

```toml
[tool.flwr.app.config]
num-server-rounds = 5           # Number of federated rounds
fraction-fit = 0.8              # Fraction of clients for training
local-epochs = 2                # Local training epochs per round
experiment-name = "ACDC_Medical_FL"
min-fit-clients = 5             # Minimum clients for training
min-available-clients = 10      # Total number of clients
```

## Advanced Usage

### Custom Experiments

Modify key parameters in `med/med/task.py`:
- `DATA_PATH`: Path to ACDC dataset
- `N_CLASSES`: Number of segmentation classes
- `PARTITION_STRATEGY`: Data distribution strategy ('iid', 'non-iid')

### Strategy Configuration

Update server strategy in `med/med/server_app.py`:
- `eta`: AdaFedAdam learning rate
- `w_impact`, `w_debt`: Fairness weights
- `lambda_val`: Adaptive loss parameter

## Dataset Setup

1. Download ACDC dataset from official source
2. Preprocess data into required format
3. Place in `data/ACDC_preprocessed/` with structure:
   ```
   ACDC_preprocessed/
   ├── ACDC_training_slices/
   ├── ACDC_training_volumes/
   └── ACDC_testing_volumes/
   ```

## Experiment Output

Each experiment generates:
- **CSV Files**: Client metrics, server metrics, convergence analysis
- **Plots**: Training curves, fairness metrics, accuracy trends  
- **Checkpoints**: Model weights at key rounds
- **Summary**: Experiment configuration and results

## Research Features

- **Comprehensive Metrics**: Dice, IoU, Hausdorff Distance, ASSD
- **Fairness Analysis**: Client contribution tracking and debt management
- **Adaptive Strategies**: Dynamic learning rate and loss adaptation
- **Statistical Validation**: Convergence tests and significance analysis

## Dependencies

All dependencies are managed in `med/pyproject.toml`:
- PyTorch 2.0+
- Flower 1.18.0+ with simulation support
- NumPy, SciPy, Pandas
- Medical imaging libraries (nibabel, scikit-image)
- Visualization tools (matplotlib, seaborn)

## Migration from Legacy

This project has been migrated from legacy `simulation.py` to modern Flower app architecture:
- ✅ Better organization and modularity
- ✅ Improved configuration management  
- ✅ No more deprecated warnings
- ✅ Compatible with latest Flower features
- ✅ Automatic post-processing and visualization

## Contributing

1. Fork the repository
2. Create feature branch
3. Follow existing code style
4. Add tests for new features
5. Submit pull request

## License

Apache 2.0 License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{federated_medical_segmentation,
  title={Federated Learning Framework for Medical Image Segmentation},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```