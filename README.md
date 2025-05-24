# Physical MedFL (Medical Federated Learning)

A comprehensive federated learning platform designed for medical imaging that supports privacy-preserving distributed training across multiple healthcare institutions while keeping sensitive patient data secure and local.

## Project Overview

Physical MedFL is an advanced framework for enabling collaborative machine learning on sensitive medical imaging data without compromising patient privacy or data security. The platform implements a federated learning approach where the model travels to the data rather than centralizing the data for training. This approach addresses critical concerns in the medical field regarding data privacy, security, and regulatory compliance.

### Key Features

- **Federated Learning Architecture**: Implements the FL paradigm where models are trained across multiple decentralized edge devices holding local data samples
  
- **Privacy Preservation**: 
  - Patient data never leaves local institutions
  - Only model parameters and gradients are shared
  - No raw medical images are transmitted over the network
  
- **Universal Medical Image Support**:
  - NIfTI (.nii, .nii.gz) - Neuroimaging Informatics Technology Initiative format
  - H5/HDF5 (.h5, .hdf5) - Hierarchical Data Format
  - DICOM (.dcm, .dicom) - Digital Imaging and Communications in Medicine
  - Common images (PNG, JPG, JPEG, TIFF)
  - NumPy arrays (.npy, .npz) - Python's numerical computing format
  - Raw binary (.raw, .bin) - Unprocessed binary data formats
  
- **Intelligent Data Loading**: The `UniversalDataLoader` class automatically detects and processes diverse medical imaging formats with format-specific optimizations

- **Advanced Data Processing**:
  - Automatic format detection
  - Standardized preprocessing across different medical image types
  - 3D to 2D slice conversion for volumetric data
  - Intensity normalization and clipping
  - Resizing to standardized dimensions
  
- **Medical-Specific Data Augmentation**:
  - Anatomically-aware augmentations
  - Conservative transformations suitable for medical imaging
  - Includes flipping, rotation, noise, brightness and contrast adjustments

- **Robust Error Handling**:
  - Graceful degradation when dependencies are missing
  - Extensive logging and debugging capabilities
  - Automated handling of edge cases in medical datasets

## Project Architecture & Components

The Physical MedFL platform is structured as a client-server architecture where:
- The server coordinates the federated learning process and aggregates model updates
- Multiple clients train the model locally on their private medical data
- Communication is secure and only model parameters are exchanged

### Project Structure

```
📂 Federated_Learning/
├── 📄 app_client.py               # Client-side application for local training on institutional data
│   ├── FLClient class             # Handles client-side model training, optimization, and communication
│   ├── ClientDataManager          # Manages local medical data access and preprocessing
│   └── ClientModelHandler         # Handles model updates, parameter merging, and local optimization
│
├── 📄 app_server.py               # Server coordination for federated model aggregation
│   ├── FLServer class             # Orchestrates the entire federated learning workflow
│   ├── ModelAggregator            # Implements secure aggregation of model updates from clients
│   └── TrainingCoordinator        # Manages federated training rounds and client participation
│
├── 📄 model_and_data_handle.py    # Core components for model management and data handling
│   ├── ModelHandler               # Abstract model management with save/load capabilities
│   ├── FederatedModelHandler      # Specialized handler for federated models 
│   ├── DataHandler                # Unified interface for medical data access and preprocessing
│   └── DatasetManager             # Handles creation and management of medical imaging datasets
│
├── 📄 universal_data_loader.py    # Universal medical image loader with format detection
│   ├── UniversalDataLoader        # Main class for loading and preprocessing multi-format medical data
│   ├── MedicalDataAugmentation    # Medical-specific data augmentation techniques
│   ├── DataFormat                 # Enumeration of supported medical data formats
│   └── create_dataloader()        # Factory function for PyTorch DataLoader creation
│
├── 📄 KAN_Model.py                # Implementation of Kolmogorov-Arnold Network model
│   ├── KANModel                   # Main KAN model implementation
│   ├── KANLayer                   # Individual KAN layer implementation
│   ├── SplineActivation           # Advanced spline-based activation functions
│   └── Optimizers                 # Specialized optimizers for KAN models
│
├── 📄 debug_training.py           # Comprehensive utilities for debugging training processes
│   ├── DebugMonitor               # Training monitoring and visualization utilities
│   ├── PerformanceProfiler        # CPU/GPU/Memory usage tracking during training
│   └── TrainingVisualizer         # Real-time training metrics visualization
│
├── 📄 check_dataset.py            # Tools to validate and inspect medical imaging datasets
│   ├── DatasetValidator           # Validates dataset structure and content
│   ├── DatasetStatistics          # Computes and reports dataset statistics
│   └── DatasetVisualizer          # Visualizes sample images from the dataset
│
├── 📄 setup_dataset.py            # Dataset preparation and preprocessing for medical imaging
│   ├── DatasetDownloader          # Downloads datasets from medical imaging repositories
│   ├── ACDCPreprocessor           # ACDC-specific preprocessing functions
│   └── DatasetSplitter            # Splits data for training, validation, and testing
│
├── 📄 debug_imports.py            # Dependency verification and environment checking
├── 📄 fix_nifti_warnings.sh       # Script to fix common NIfTI-related warnings
├── 📄 test_client.sh              # Script for testing client functionality
├── 📄 pyproject.toml              # Project configuration, dependencies, and build settings
├── 📄 .gitignore                  # Git ignore configurations for version control
│
└── 📂 ACDC/                       # ACDC cardiac MRI dataset (not included in repository)
    ├── 📂 database/
    │   ├── 📂 training/           # Training dataset with 100 patient cases
    │   │   └── 📂 patient001-100/ # Individual patient directories with cardiac MRI sequences
    │   └── 📂 testing/            # Testing dataset with 50 patient cases
    │       └── 📂 patient101-150/ # Test patient directories with cardiac MRI sequences
    └── 📄 MANDATORY_CITATION.md   # Citation information for the ACDC dataset
```

## Technical Architecture & Components

Physical MedFL implements a sophisticated federated learning architecture with several key technical components working together seamlessly.

### Core Components

#### 1. Server-Client Federated Infrastructure

**🔄 Federated Learning Server (`app_server.py`)**
- **Global Model Management**: Maintains the global model state and coordinates updates
- **Aggregation Strategies**: Implements FedAvg, FedProx, and custom aggregation algorithms
- **Round-Robin Coordination**: Orchestrates training rounds across multiple institutions
- **Client Management**: Handles client registration, authentication, and communication
- **Privacy Preservation**: Ensures no raw data is shared during the training process
- **Asynchronous Processing**: Supports asynchronous client updates without blocking

**🏥 Client Applications (`app_client.py`)**
- **Local Training**: Executes model training on local institutional data
- **Differential Privacy**: Implements ε-differential privacy methods to protect patient data
- **Secure Communication**: Establishes encrypted channels for model parameter updates
- **Resource Management**: Optimizes GPU/CPU usage based on available institution resources
- **Auto-resumption**: Supports continuation of training from checkpoint after interruptions
- **Data Management**: Handles local medical imaging data access and preprocessing

#### 2. Advanced Data Processing Pipeline

**🩻 Universal Medical Imaging Support (`universal_data_loader.py`)**
- **Format Auto-detection**: Intelligently identifies and loads various medical imaging formats
- **Vectorized Processing**: Implements high-performance data transformations using NumPy/PyTorch
- **Adaptive Preprocessing**: Format-specific preprocessing optimizations for medical images
- **Memory Efficiency**: Streaming interfaces for large volumetric medical datasets
- **Data Augmentation**: Medical-specific augmentations that preserve diagnostic features
- **Batching and Caching**: Performance-optimized data loading with prefetching capabilities
- **Multi-threading**: Parallel data loading and preprocessing for improved performance

**🛠️ Dataset Management Tools**
- **Dataset Preparation (`setup_dataset.py`)**: Standardizes and organizes medical imaging data
- **Validation Utilities (`check_dataset.py`)**: Ensures data quality and consistency
- **Visualization Tools**: Generates visualizations of sample images and segmentations

#### 3. Model Implementation & Management

**🧠 Advanced Model Architecture**
- **Model Management (`model_and_data_handle.py`)**: Unified interface for model operations
- **KAN Implementation (`KAN_Model.py`)**: Kolmogorov-Arnold Networks for medical imaging
  - **Theoretical Advantages**: Universal approximation with adaptive complexity
  - **Spline Activations**: Smoother gradient flow for better convergence
  - **Interpretability**: Enhanced model explainability compared to traditional networks
  - **Parameter Efficiency**: Reduced parameter count for comparable performance

**⚙️ Training & Optimization**
- **Custom Optimizers**: Specialized optimizers for federated medical imaging tasks
- **Adaptive Learning Rates**: Learning rate policies optimized for federated scenarios
- **Regularization Methods**: L1/L2 and medical-specific regularization techniques

#### 4. Development & Debugging Tools

**🔍 Comprehensive Debugging Suite**
- **Training Debugging (`debug_training.py`)**: Advanced debugging for training processes
  - **Performance Profiling**: Memory and compute resource utilization tracking
  - **Gradient Flow Analysis**: Visualization of gradient magnitudes across layers
  - **Activation Visualization**: Feature map visualization capabilities
  - **Training Metrics**: Real-time monitoring and logging of training metrics
  
- **Environment Management**
  - **Dependency Verification (`debug_imports.py`)**: Ensures all required packages are installed
  - **NIfTI Warning Resolution (`fix_nifti_warnings.sh`)**: Fixes common NIfTI-related warnings
  - **Client Testing (`test_client.sh`)**: Validates client functionality in isolation

## Detailed API Reference

### UniversalDataLoader

The heart of the system's medical imaging capabilities, enabling seamless processing of diverse medical imaging formats.

```python
class UniversalDataLoader:
    def __init__(self, 
                target_size: Tuple[int, int] = (256, 256),
                num_classes: int = 4,
                normalize: bool = True,
                clip_range: Optional[Tuple[float, float]] = None):
        """
        Initialize universal data loader.
        
        Args:
            target_size: Target image size (height, width)
            num_classes: Number of segmentation classes
            normalize: Whether to normalize images to [0, 1]
            clip_range: Optional intensity clipping range
        """
        
    def detect_format(self, path: str) -> str:
        """Auto-detect data format from file/directory structure."""
        
    def load_data(self, path: str, 
                 max_samples: Optional[int] = None, 
                 format_hint: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Universal data loading with automatic format detection.
        
        Args:
            path: Path to data directory or file
            max_samples: Maximum number of samples to load
            format_hint: Optional format hint to skip detection
            
        Returns:
            Tuple of (images, masks) where masks can be None
        """
```

### Federated Learning Components

Core server and client components for implementing the federated learning protocol.

```python
class FLServer:
    def __init__(self, model_path=None, port=8000, clients_required=2, 
                 rounds=10, epochs_per_round=1, aggregation_method='fedavg'):
        """
        Initialize federated learning server.
        
        Args:
            model_path: Path to initial model parameters
            port: Port to listen for client connections
            clients_required: Minimum number of clients needed to start training
            rounds: Number of federated training rounds
            epochs_per_round: Client local epochs per round
            aggregation_method: Method for aggregating client updates
        """
    
    def start(self):
        """Start the federated learning server and wait for client connections."""
        
    def aggregate_models(self, client_models):
        """Aggregate client model updates using the selected method."""


class FLClient:
    def __init__(self, server_address, data_path, client_id=None):
        """
        Initialize federated learning client.
        
        Args:
            server_address: Address of the FL server
            data_path: Path to local medical imaging data
            client_id: Unique identifier for this client
        """
        
    def connect(self):
        """Connect to the federated learning server."""
        
    def train_local_model(self, model, epochs):
        """Train the model on local data for specified epochs."""
        
    def send_model_update(self, model):
        """Send local model update to server."""
```

## Installation Guide

1. Clone the repository:
```bash
git clone https://github.com/QuocKhanhLuong/Physical_MedFL.git
cd Physical_MedFL
```

2. Create a virtual environment with Python 3.10 or newer:
```bash
python3 -m venv .venv_py310
source .venv_py310/bin/activate  # On Unix/macOS
# OR
.\.venv_py310\Scripts\activate.bat  # On Windows CMD
# OR
.\.venv_py310\Scripts\Activate.ps1  # On Windows PowerShell
```

3. Install the dependencies:
```bash
# Install base dependencies
pip install -e .

# Install optional dependencies for full functionality
pip install -e ".[all]"  # Installs all optional dependencies
# OR
pip install -e ".[dicom]"  # DICOM support only
pip install -e ".[nifti]"  # NIfTI support only
pip install -e ".[viz]"    # Visualization support only
```

4. Verify installation:
```bash
python debug_imports.py
```

## Usage Examples

### Setting Up the Federated Learning Environment

#### Running the Server

Start the federated learning server with default settings:

```bash
python app_server.py
```

With custom configuration:

```bash
python app_server.py --port 8080 --clients 5 --rounds 20 --aggregation fedprox --epsilon 0.1
```

Parameters:
- `--port`: Port for client connections (default: 8000)
- `--clients`: Minimum clients required to begin training (default: 2)
- `--rounds`: Number of federated training rounds (default: 10)
- `--aggregation`: Aggregation method (fedavg, fedprox, etc.)
- `--epsilon`: Differential privacy parameter (default: 0.0)

#### Running a Client

Start a federated learning client to train on local data:

```bash
python app_client.py --server localhost:8000 --data /path/to/medical/data --id hospital1
```

Parameters:
- `--server`: Server address (default: localhost:8000)
- `--data`: Path to local medical imaging data
- `--id`: Unique client identifier
- `--gpu`: GPU device to use (default: 0)
- `--batch`: Batch size (default: 16)

### Working with Medical Imaging Data

Loading and preprocessing medical imaging data:

```python
from universal_data_loader import UniversalDataLoader, create_dataloader

# Initialize loader
loader = UniversalDataLoader(
    target_size=(256, 256),  # Target dimensions
    num_classes=4,           # For segmentation tasks
    normalize=True,          # Normalize pixel values to [0,1]
    clip_range=(-1000, 1000) # HU range for CT images
)

# Load data (auto-detects format)
images, masks = loader.load_data(
    path="patient_scans/",
    max_samples=100  # Optionally limit sample count
)

# Create PyTorch DataLoader with augmentation
dataloader = create_dataloader(
    images=images,
    masks=masks,
    batch_size=16,
    shuffle=True,
    augment=True  # Enable medical-appropriate augmentations
)

# Use in training loop
for batch_imgs, batch_masks in dataloader:
    # Model training code here
    pass
```

### ACDC Dataset Setup

The project uses the ACDC (Automated Cardiac Diagnosis Challenge) dataset, which contains cardiac MRI images. Due to size constraints and licensing requirements, the dataset must be downloaded separately.

To set up the dataset:

```bash
# Download and prepare the ACDC dataset
./setup_dataset.py --dataset acdc --destination ./ACDC

# Verify dataset integrity
python check_dataset.py --path ./ACDC --visualize
```

## Benchmarks

| Dataset | Model | Federated Clients | Dice Score | Training Time |
|---------|-------|-------------------|------------|---------------|
| ACDC    | KAN   | 4                 | 0.92       | 4.2h          |
| ACDC    | UNet  | 4                 | 0.89       | 5.8h          |
| ACDC    | KAN   | 8                 | 0.90       | 2.7h          |
| ACDC    | UNet  | 8                 | 0.87       | 3.9h          |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Physical MedFL in your research, please cite:

```bibtex
@software{luong2025physical,
  author = {Luong, Quoc Khanh},
  title = {Physical MedFL: Privacy-Preserving Federated Learning for Medical Imaging},
  year = {2025},
  url = {https://github.com/QuocKhanhLuong/Physical_MedFL}
}
```

## Contact & Contributions

**Project Lead:** Quoc Khanh Luong

To contribute to this project:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For issues, questions, or collaboration opportunities, please open an issue on GitHub or contact the project maintainer.