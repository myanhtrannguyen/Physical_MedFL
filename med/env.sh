#!/bin/bash
# Script quản lý môi trường Conda cho dự án Federated Learning

set -e # Thoát ngay khi có lỗi

# --- CẤU HÌNH ---
# Tên môi trường Conda bạn muốn tạo
ENV_NAME="fl_env"
# Phiên bản Python bạn muốn sử dụng
PYTHON_VERSION="3.11" 

# In màu cho đẹp
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
print_status() { echo -e "${BLUE}[INFO] $1${NC}"; }
print_success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }

# --- HÀM SETUP ---
setup_environment() {
    print_status "Kiểm tra môi trường Conda '$ENV_NAME'..."
    # Xóa môi trường cũ nếu tồn tại để cài lại cho sạch
    if conda env list | grep -q "$ENV_NAME"; then
        print_status "Môi trường đã tồn tại. Đang xóa để tạo mới..."
        conda env remove --name "$ENV_NAME" -y
    fi

    print_status "Đang tạo môi trường Conda '$ENV_NAME' với Python ${PYTHON_VERSION}..."
    conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y
    
    print_status "Đang cài đặt các thư viện từ pyproject.toml..."
    # Dùng `conda run` để chạy lệnh pip trong môi trường vừa tạo
    conda run -n "$ENV_NAME" pip install -e .
    
    print_success "Hoàn tất! Môi trường '$ENV_NAME' đã sẵn sàng."
    print_status "Hãy khởi động lại VS Code, sau đó chọn interpreter tên là '$ENV_NAME'."
    print_status "Trong terminal mới, hãy chạy 'conda activate $ENV_NAME' để bắt đầu."
}

# --- LOGIC CHÍNH ---
if [[ "$1" == "setup" ]]; then
    setup_environment
else
    echo "Sử dụng:"
    echo "  ./env.sh setup    - Để tạo môi trường Conda và cài đặt thư viện."
    echo "  conda activate $ENV_NAME - Để kích hoạt môi trường trong terminal."
fi