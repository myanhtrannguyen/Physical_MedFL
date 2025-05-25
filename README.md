# Federated Learning for Medical Image Segmentation

A comprehensive federated learning framework for medical image segmentation using the ACDC (Automated Cardiac Diagnosis Challenge) dataset. This project implements robust federated learning with support for multiple model architectures and data partitioning strategies.

## 🏗️ Project Structure

```
FEDERATED_LEARNING/
│
├── pyproject.toml             # ✅ Flower project config
│
├── data/
│   ├── raw/                   # Untouched raw data (ACDC dataset)
│   ├── processed/             # Cleaned, normalized data
│   └── partitions/            # Federated splits per client
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Cleaning, normalization, augmentation
│   │   ├── partitioning.py    # IID/non-IID splits for FL
│   │   ├── dataset.py         # Custom Dataset classes
│   │   └── loader.py          # DataLoader utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kan_model.py       # KAN model implementation
│   │   ├── mlp_model.py       # MLP/U-Net model implementation
│   │   └── model_factory.py   # Factory pattern for models
│   │
│   ├── fl_core/
│   │   ├── __init__.py
│   │   ├── app_client.py      # FlowerClient logic
│   │   ├── app_server.py      # Server strategy logic
│   │   ├── aggregation.py     # Aggregation strategies
│   │   └── communication.py   # Communication protocols/utils
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py            # Reproducibility utilities
│   │   └── logger.py          # Logging utilities
│   │
│   └── experiment/
│       ├── config.yaml        # All experiment configurations
│       ├── run_fl.py          # Main FL pipeline script
│       └── run_centralized.py # Centralized baseline
│
├── scripts/
│   ├── prepare_data.py        # Data preprocessing & partitioning
│   └── visualize_data.py      # Data exploration/EDA
│
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Federated_Learning

# Create virtual environment
python -m venv .venv_py310
source .venv_py310/bin/activate  # On Windows: .venv_py310\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Data Preparation

```bash
# Validate data structure
python scripts/prepare_data.py --validate

# Create federated partitions
python scripts/prepare_data.py --partition

# Run all preparation steps
python scripts/prepare_data.py --all
```

### 3. Run Federated Learning

```bash
# Using Flower simulation
flower-simulation --app . --num-supernodes 5

# Or run individual components
flower-server --app src.fl_core.app_server:app
flower-client --app src.fl_core.app_client:app
```

## 📊 Dataset

This project uses the **ACDC (Automated Cardiac Diagnosis Challenge)** dataset for cardiac image segmentation:

- **Classes**: 4 (Background, Right Ventricle, Myocardium, Left Ventricle)
- **Image Size**: 256×256 (after preprocessing)
- **Format**: NIfTI (.nii files)
- **Frames**: End-diastolic (ED) and End-systolic (ES)

### Data Structure
```
data/raw/ACDC/
├── patient001/
│   ├── patient001_frame01.nii      # ED image
│   ├── patient001_frame01_gt.nii   # ED ground truth
│   ├── patient001_frame12.nii      # ES image
│   └── patient001_frame12_gt.nii   # ES ground truth
├── patient002/
└── ...
```

## 🔧 Configuration

All experiment settings are configured in `src/experiment/config.yaml`:

```yaml
# Model configuration
model:
  type: "unet"  # Options: unet, kan, mlp
  n_channels: 1
  n_classes: 4
  dropout_rate: 0.1

# Federated learning
federated:
  num_clients: 5
  num_rounds: 10
  partition_type: "iid"  # Options: iid, non_iid, pathological
  alpha: 0.5  # For non-IID Dirichlet distribution

# Training
training:
  batch_size: 8
  learning_rate: 0.001
  optimizer: "adam"
```

## 🏛️ Model Architectures

### 1. U-Net (RobustMedVFL_UNet)
- **Purpose**: Medical image segmentation
- **Architecture**: Encoder-decoder with skip connections
- **Features**: Dropout, batch normalization, residual connections

### 2. KAN (Kolmogorov-Arnold Networks)
- **Purpose**: Alternative to traditional MLPs
- **Features**: Learnable activation functions, spline-based
- **Use Case**: Experimental architecture for comparison

### 3. MLP (Multi-Layer Perceptron)
- **Purpose**: Baseline comparison
- **Architecture**: Simple feedforward network
- **Features**: Configurable hidden layers, dropout

## 🔄 Federated Learning Features

### Data Partitioning Strategies
1. **IID (Independent and Identically Distributed)**
   - Equal distribution across clients
   - Balanced data splits

2. **Non-IID (Dirichlet Distribution)**
   - Configurable heterogeneity with α parameter
   - Realistic federated scenarios

3. **Pathological Non-IID**
   - Each client has only few classes
   - Extreme heterogeneity simulation

### Aggregation Strategies
- **FedAvg**: Weighted averaging of model parameters
- **Custom strategies**: Extensible framework for new methods

## 📈 Evaluation Metrics

- **Dice Coefficient**: Overlap-based similarity
- **IoU (Intersection over Union)**: Jaccard index
- **Accuracy**: Pixel-wise accuracy
- **Precision & Recall**: Class-specific metrics

## 🛠️ Development

### Adding New Models
1. Implement model in `src/models/`
2. Add factory method in `model_factory.py`
3. Update configuration options

### Adding New Datasets
1. Create dataset class in `src/data/dataset.py`
2. Implement data loader in `src/data/loader.py`
3. Update preprocessing pipeline

### Custom Federated Strategies
1. Extend base strategy in `src/fl_core/`
2. Implement aggregation logic
3. Update server configuration

## 📝 Logging and Monitoring

- **Structured Logging**: Consistent format across components
- **Client-specific Logs**: Individual client tracking
- **Experiment Tracking**: Configuration and metrics logging
- **Progress Monitoring**: Real-time training progress

## 🔬 Reproducibility

- **Seed Management**: Consistent random seeds
- **Deterministic Operations**: Reproducible results
- **Configuration Versioning**: Experiment tracking
- **Environment Isolation**: Virtual environment setup

## 📚 Key Features

- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Multiple Models**: U-Net, KAN, MLP support
- ✅ **Flexible Partitioning**: IID/Non-IID data splits
- ✅ **Comprehensive Logging**: Detailed experiment tracking
- ✅ **Configuration Management**: YAML-based settings
- ✅ **Data Validation**: Automated structure checking
- ✅ **Reproducible Results**: Seed management
- ✅ **Extensible Framework**: Easy to add new components

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ACDC Challenge**: For providing the cardiac segmentation dataset
- **Flower Framework**: For federated learning infrastructure
- **PyTorch**: For deep learning capabilities
- **Medical Imaging Community**: For advancing healthcare AI

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review example configurations in `src/experiment/`

---

**Note**: This is a research project for medical image segmentation using federated learning. Ensure proper data handling and privacy compliance when working with medical data.