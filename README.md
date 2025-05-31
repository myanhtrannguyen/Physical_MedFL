# Federated Learning for Medical Image Segmentation

This project implements a federated learning system for medical image segmentation, primarily designed for datasets like ACDC (cardiac) and BraTS 2020 (brain tumor). It utilizes the Flower framework for federated learning orchestration.

## Prerequisites

*   Python 3.10 or newer
*   Pip (Python package installer)
*   Git (for cloning the repository)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd <repository_directory_name> 
    ```
    *(Replace `<your_repository_url>` with the actual URL of your Git repository and `<repository_directory_name>` with the name of the cloned folder, e.g., `Federated_Learning`)*

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv_py310
    source .venv_py310/bin/activate
    ```
    *(On Windows, use `.venv_py310\Scripts\activate` to activate)*

3.  **Install Dependencies:**
    This project uses `pyproject.toml` to manage dependencies. Install them using:
    ```bash
    pip install .
    ```
    This command will install all necessary packages, including Flower, PyTorch, Nibabel, etc., as defined in `pyproject.toml`.

4.  **Download Datasets:**
    The system is designed to work with data organized as follows. Create these directories if they don't exist:

    *   **ACDC Dataset:**
        *   Download the ACDC dataset (e.g., from [ACDC Challenge website](https://www.creatis.insa-lyon.fr/Challenge/acdc/)).
        *   Extract and place the training data (patient folders) into `data/raw/ACDC/training/`. The expected structure is:
            ```
            Federated_Learning/
            └── data/
                └── raw/
                    └── ACDC/
                        └── training/
                            ├── patient001/
                            │   ├── patient001_frame01.nii.gz
                            │   ├── patient001_frame01_gt.nii.gz
                            │   └── ... (other frames and patient info files)
                            ├── patient002/
                            │   └── ...
                            └── ...
            ```
    *   **BraTS 2020 Dataset (Preprocessed H5 files):**
        *   If you are using BraTS 2020 data preprocessed into H5 files (each file representing a 2D slice with multiple modalities and a mask), place them into `data/raw/BraTS2020_training_data/content/data/`.
        *   The dataset loader expects H5 files named in a pattern like `volume_<VOL_ID>_slice_<SLICE_ID>.h5`.
            ```
            Federated_Learning/
            └── data/
                └── raw/
                    └── BraTS2020_training_data/
                        └── content/
                            └── data/
                                ├── volume_0_slice_75.h5
                                ├── volume_0_slice_76.h5
                                └── ...
            ```
    *(Note: The exact paths used by the scripts to find these datasets are typically configured within your main simulation script, where `base_data_dirs` is passed to `get_client_dataloader_direct` in `src/data/flower_utils.py`.)*

5.  **Configuration:**
    *   Federated learning strategy parameters (e.g., number of rounds, FedAdam parameters) are in `config.toml`.
    *   The number of clients, dataset types per client, and specific dataset arguments (like `frames` for ACDC or `slice_range` for BraTS) are usually set in your main Python script that launches the Flower simulation. This script will use `src/data/flower_utils.py` to prepare data for each client.

## Running the Federated Learning Simulation

1.  **Activate the Virtual Environment (if not already active):**
    ```bash
    source .venv_py310/bin/activate
    ```

2.  **Run the Main Simulation Script:**
    You will need a Python script to define the Flower simulation, configure clients, and start the server. Let's assume you have a script like `src/run_federated_simulation.py` ( **Please create or identify your actual main script** ).

    Example of how you might run it:
    ```bash
    python src/run_federated_simulation.py
    ```
    This script should:
    *   Define `TOTAL_NUM_CLIENTS`.
    *   For each client:
        *   Determine which datasets (`dataset_types_for_client`) it should use.
        *   Define `base_data_dirs` pointing to your `data/raw/ACDC/training` and `data/raw/BraTS2020_training_data/content/data` (or wherever you placed them).
        *   Instantiate `MedicalImagePreprocessor` and `DataAugmentation`.
        *   Call `get_client_dataloader_direct` from `src/data/flower_utils.py` to get the client's DataLoader.
        *   Define a Flower `NumPyClient` or `Client` that uses this DataLoader for training and evaluation.
    *   Start the Flower simulation using `fl.simulation.start_simulation()`.

    Logs and results will typically be saved to the `logs/` directory.

## Project Structure Overview

```
Federated_Learning/
├── .venv_py310/            # Python virtual environment
├── config.toml             # Flower strategy configuration
├── data/
│   └── raw/                # Root for raw ACDC and BraTS datasets
├── logs/                   # Directory for experiment outputs (models, metrics)
├── src/
│   ├── data/               # Data loading, preprocessing, augmentation, dataset classes, Flower data utilities
│   │   ├── dataset.py
│   │   └── flower_utils.py
│   ├── models/             # Segmentation model definitions (e.g., UNet)
│   ├── experiment/         # Training, evaluation, and experiment logic
│   ├── utils/              # Common utility functions
│   └── run_federated_simulation.py # EXAMPLE: Your main script to start Flower (you need to create/identify this)
├── .gitignore
├── LICENSE
├── pyproject.toml          # Defines project dependencies and metadata
└── README.md               # This file
```

## Customization

*   **Federated Strategy:** Modify `config.toml` and the strategy definition in your main simulation script.
*   **Number of Clients/Rounds:** Adjust in `config.toml` and your main simulation script.
*   **Model Architecture:** Modify or add models in `src/models/`.
*   **Data Augmentation/Preprocessing:** Configure `MedicalImagePreprocessor` and `DataAugmentation` instances passed to data loaders in `src/data/`.
*   **Client Data Distribution:** The logic in `src/data/dataset.py` (specifically in `ACDCUnifiedDataset` and `BraTS2020UnifiedDataset`) handles how data is split among clients based on `client_id` and `total_num_clients`. This is utilized by `get_client_dataloader_direct` in `src/data/flower_utils.py`.

## Troubleshooting

*   **`ModuleNotFoundError`:** Ensure your virtual environment is active and you've run `pip install .` successfully. If importing local modules from `src` (e.g., `from src.data import ...`), make sure Python's path recognizes the `src` directory or run your main script from the project root.
*   **Data Loading Issues (`FileNotFoundError`, empty datasets):**
    *   Verify that the dataset paths in `base_data_dirs` within your main simulation script accurately point to the locations of your ACDC and BraTS data.
    *   Check that the file and directory structure within your `data/raw/` matches the expectations outlined in the "Download Datasets" section.
    *   Look for warnings from `logger` in the console output, which might indicate issues finding or assigning data to clients.
*   **PyTorch/CUDA Errors:** If using a GPU, confirm that PyTorch was installed with CUDA support and that your NVIDIA drivers are compatible. `pin_memory=torch.cuda.is_available()` in `get_client_dataloader_direct` helps with GPU performance.

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