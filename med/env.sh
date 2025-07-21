#!/bin/bash

# Federated Learning Environment Management Script (Corrected)
# Usage:
#   ./env.sh setup    - Setup virtual environment and install dependencies
#   source env.sh     - Activate existing environment

set -e  # Exit on any error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
PYTHON_CMD="python3"

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO] $1${NC}"; }
print_success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
print_warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
print_error() { echo -e "${RED}[ERROR] $1${NC}"; }

setup_environment() {
    print_status "Setting up Federated Learning environment..."
    
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run this script from the 'med' directory."
        return 1
    fi
    
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_error "Python 3 not found. Please install it first."
        return 1
    fi
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Existing virtual environment found. Removing..."
        rm -rf "$VENV_DIR"
    fi
    
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    source "$VENV_DIR/bin/activate"
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_status "Installing Flower app and dependencies from pyproject.toml..."
    # <<< FIX: Run pip install directly from the project root (med/) >>>
    pip install -e .
    
    print_success "Environment setup completed!"
    print_status "To activate in a new terminal, run: source env.sh"
}

activate_environment() {
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        print_error "Virtual environment not found. Run './env.sh setup' first."
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated."
    echo "You can now run the simulation using: flwr run"
}

# Main script logic
if [ "$0" = "${BASH_SOURCE[0]}" ]; then
    # Script is being executed directly
    case "${1:-help}" in
        "setup") setup_environment ;;
        *)
            echo "Usage: ./env.sh setup"
            echo "To activate the environment, use: source env.sh"
            ;;
    esac
else
    # Script is being sourced
    activate_environment
fi