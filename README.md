# 🏥 Federated Learning for Medical Image Segmentation

**A complete research framework for federated learning on ACDC cardiac dataset with state-of-the-art metrics tracking and adaptive loss functions.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Flower](https://img.shields.io/badge/Flower-1.18%2B-green)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)

## 📋 Table of Contents

- [🔧 System Requirements](#-system-requirements)
- [💻 Installation Guide](#-installation-guide)
  - [Windows Setup](#windows-setup)
  - [macOS Setup](#macos-setup)
- [📂 Project Setup](#-project-setup)
- [🔍 Data Preparation](#-data-preparation)
- [🚀 Running Experiments](#-running-experiments)
- [📊 Viewing Results](#-viewing-results)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [📖 Advanced Usage](#-advanced-usage)

---

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 or macOS 10.15+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.8 or higher

### Recommended Requirements
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with CUDA support (optional but faster)
- **Storage**: 20GB+ for full datasets

---

## 💻 Installation Guide

### Windows Setup

#### Step 1: Install Python
1. **Download Python**: Go to [python.org](https://python.org/downloads/) and download Python 3.8+
2. **Install**: Run the installer with these options:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install pip"
   - Choose "Install Now"

3. **Verify Installation**:
   ```bash
   # Open Command Prompt (cmd) and run:
   python --version
   pip --version
   ```

#### Step 2: Install Git
1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Install with default settings
3. Verify: `git --version` in Command Prompt

#### Step 3: Clone Repository
```bash
# Open Command Prompt and navigate to desired folder
cd C:\Users\YourName\Documents
git clone https://github.com/your-repo/Federated_Learning.git
cd Federated_Learning
```

### macOS Setup

#### Step 1: Install Prerequisites
1. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

2. **Install Homebrew** (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Install Python**:
   ```bash
   brew install python@3.11
   ```

4. **Install Git**:
   ```bash
   brew install git
   ```

#### Step 2: Clone Repository
```bash
# Open Terminal and navigate to desired folder
cd ~/Documents
git clone https://github.com/your-repo/Federated_Learning.git
cd Federated_Learning
```

---

## 📂 Project Setup

### Method 1: Automatic Setup (Recommended)

#### For macOS/Linux:
```bash
# Make env.sh executable and run setup
chmod +x med/env.sh
cd med
./env.sh setup
```

#### For Windows:
```bash
# Navigate to med directory
cd med

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .
```

### Method 2: Manual Setup

#### Step 1: Create Virtual Environment
```bash
# Navigate to project folder
cd Federated_Learning/med

# Create virtual environment
python -m venv venv
```

#### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install the project
pip install -e .
```

#### Step 4: Verify Installation
```bash
# Check if Flower is installed correctly
flwr --version

# Test import
python -c "import torch; import flwr; print('✅ All dependencies installed successfully!')"
```

---

## 🔍 Data Preparation

### Option 1: Use Sample Data (Quick Start)
```bash
# Download and extract sample ACDC data (placeholder - replace with actual download)
# For demo purposes, create sample data structure:

cd Federated_Learning
mkdir -p data/ACDC_preprocessed/ACDC_training_slices
mkdir -p data/ACDC_preprocessed/ACDC_training_volumes  
mkdir -p data/ACDC_preprocessed/ACDC_testing_volumes

# Note: Add your actual .h5 files to these directories
```

### Option 2: Full ACDC Dataset
1. **Register** at [ACDC Challenge website](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
2. **Download** the full dataset
3. **Preprocess** using provided scripts:
   ```bash
   # Run preprocessing script (if available)
   python scripts/preprocess_acdc.py --input raw_data/ --output data/ACDC_preprocessed/
   ```

### Expected Data Structure
```
Federated_Learning/
├── data/
│   └── ACDC_preprocessed/
│       ├── ACDC_training_slices/    # 2D slice files (.h5)
│       ├── ACDC_training_volumes/   # 3D volume files (.h5)
│       └── ACDC_testing_volumes/    # Test data (.h5)
```

---

## 🚀 Running Experiments

### Step 1: Activate Environment

**Every time you want to run the project:**

**Windows:**
```bash
cd Federated_Learning\med
venv\Scripts\activate
```

**macOS/Linux:**
```bash
cd Federated_Learning/med
source venv/bin/activate
# or use: source env.sh
```

### Step 2: Run Basic Experiment
```bash
# Run with default settings (5 clients, 2 rounds)
flwr run .
```

### Step 3: Custom Configuration
```bash
# Edit configuration first
nano pyproject.toml  # or use any text editor

# Run with custom settings
flwr run . --run-config num-server-rounds=5 fraction-fit=0.8
```

### Step 4: Monitor Progress
```bash
# The terminal will show real-time progress:
# ✅ Starting Flower ServerApp
# 📊 Round 1: Training 4 clients...
# 📈 Server evaluation: Loss=0.89, Accuracy=0.75
# ⏰ Round 1 completed in 180.5s
```

---

## 📊 Viewing Results

### Automatic Export
After training completes, results are automatically saved to:
```
research_exports/[experiment_name]/
├── server_aggregation_metrics.csv    # Server metrics per round
├── server_evaluation_metrics.csv     # Centralized evaluation  
├── client_detailed_metrics.csv       # Client training details
├── convergence_analysis.csv          # Convergence tracking
├── experiment_summary.json           # Overall performance
└── faup_final_debts.csv              # Fairness analysis
```

### View Results
```bash
# Navigate to results
cd research_exports

# List experiments
ls -la

# Open latest experiment folder
cd [latest_experiment_folder]

# View summary
cat experiment_summary.json

# Or open CSV files in Excel/LibreOffice
```

### Quick Performance Check
```bash
# Check final performance
python -c "
import json
with open('research_exports/[experiment_name]/experiment_summary.json') as f:
    summary = json.load(f)
    print(f'Final Accuracy: {summary.get(\"final_accuracy\", 0):.4f}')
    print(f'Best Dice Score: {summary.get(\"best_fg_dice\", 0):.4f}')
    print(f'Convergence: {summary.get(\"convergence_achieved\", False)}')
"
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Import Error: No module named 'flwr'**
```bash
# Solution: Reinstall in development mode
pip uninstall flwr
pip install flwr[simulation]
pip install -e .
```

#### 2. **CUDA/GPU Issues**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Force CPU-only mode (add to config)
# device = "cpu"
```

#### 3. **Memory Issues**
```bash
# Reduce batch size in pyproject.toml:
[tool.flwr.app.config]
batch-size = 4  # Default is 8

# Or reduce number of workers:
num-workers = 1  # Default is 2
```

#### 4. **Port Already in Use**
```bash
# Find and kill processes using port
# Windows:
netstat -ano | findstr :8080
taskkill /PID <process_id> /F

# macOS/Linux:
lsof -ti:8080 | xargs kill -9
```

#### 5. **Data Loading Errors**
```bash
# Verify data structure
ls -la data/ACDC_preprocessed/

# Check H5 files
python -c "
import h5py
import os
data_dir = 'data/ACDC_preprocessed/ACDC_training_slices'
files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
print(f'Found {len(files)} H5 files')
if files:
    with h5py.File(os.path.join(data_dir, files[0]), 'r') as f:
        print('Keys:', list(f.keys()))
"
```

### Performance Tips

#### 1. **Speed Up Training**
```toml
# In pyproject.toml:
[tool.flwr.app.config]
num-workers = 4        # Increase if you have more CPU cores
batch-size = 16        # Increase if you have more RAM
local-epochs = 1       # Reduce for faster rounds
```

#### 2. **Reduce Memory Usage**
```toml
[tool.flwr.app.config]
batch-size = 4         # Smaller batches
num-workers = 1        # Less parallel loading
min-available-clients = 3  # Fewer clients
```

#### 3. **Quick Testing**
```toml
[tool.flwr.app.config]
num-server-rounds = 2  # Just 2 rounds for testing
min-fit-clients = 2    # Minimum clients
fraction-fit = 0.6     # Train fewer clients per round
```

---

## 📖 Advanced Usage

### Custom Experiment Configuration
```bash
# Create custom config file
cp pyproject.toml my_experiment.toml

# Edit your settings
nano my_experiment.toml

# Run with custom config
flwr run . --config my_experiment.toml
```

### Key Configuration Options
```toml
[tool.flwr.app.config]
# Experiment Settings
experiment-name = "MyExperiment_2024"
num-server-rounds = 10
fraction-fit = 0.8

# Client Settings  
local-epochs = 3
batch-size = 8
min-fit-clients = 3
min-available-clients = 5

# Data Settings
partition-strategy = "non-iid"  # or "iid", "pathology-skew"
alpha = 0.5                     # Non-IID parameter
val-ratio = 0.2                 # Validation split

# Performance Settings
num-workers = 2                 # Data loading workers
learning-rate = 0.001           # Learning rate
```

### Running Multiple Experiments
```bash
# Create experiment batch script
cat > run_experiments.sh << 'EOF'
#!/bin/bash
experiments=("iid" "non-iid" "pathology-skew")
for strategy in "${experiments[@]}"; do
    echo "Running $strategy experiment..."
    flwr run . --run-config \
        experiment-name="ACDC_${strategy}" \
        partition-strategy="$strategy" \
        num-server-rounds=5
    sleep 10
done
EOF

chmod +x run_experiments.sh
./run_experiments.sh
```

### Performance Monitoring
```bash
# Real-time GPU monitoring (if using GPU)
nvidia-smi -l 1

# System resource monitoring
# Windows: Task Manager
# macOS: Activity Monitor  
# Linux: htop or top
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Citation

If you use this framework in your research, please cite:

```bibtex
@software{federated_medical_segmentation_2024,
  title={Federated Learning Framework for Medical Image Segmentation},
  author={Your Team},
  year={2024},
  url={https://github.com/your-repo/Federated_Learning}
}
```

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/Federated_Learning/issues)
- **Documentation**: [Project Wiki](https://github.com/your-repo/Federated_Learning/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/Federated_Learning/discussions)

---

### ✨ Quick Start Summary

```bash
# 1. Clone repository
git clone https://github.com/your-repo/Federated_Learning.git
cd Federated_Learning/med

# 2. Setup environment  
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .

# 3. Run experiment
flwr run .

# 4. View results
ls research_exports/
```

**🎉 You're ready to start federated learning research!**