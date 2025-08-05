import torch
import torch.nn as nn
def quantum_noise_injection(features, T=1.25, pauli_prob={'X': 0.00096, 'Y': 0.00096, 'Z': 0.00096, 'None': 0.99712}):
    """
    Áp dụng nhiễu lượng tử dựa trên cơ chế Pauli Noise Injection cho dữ liệu ảnh MRI.
    
    Args:
        features (torch.Tensor): Tensor đầu vào dạng (batch_size, channels, height, width).
        T (float): Hệ số nhiễu, thường trong khoảng [0.5, 1.5].
        pauli_prob (dict): Phân phối xác suất cho các cổng Pauli (X, Y, Z, None).
    
    Returns:
        torch.Tensor: Tensor đầu ra với nhiễu lượng tử được áp dụng.
    """
    # Chuyển features sang kiểu float
    features_float = features.float()
    
    # Kiểm tra kích thước tensor
    if features_float.dim() < 4 or features_float.size(2) < 2 or features_float.size(3) < 2:
        return features_float

    try:
        # Đảm bảo tensor ở trên thiết bị đúng
        device = features_float.device
        
        # Chuẩn hóa xác suất Pauli với hệ số T
        scaled_prob = {
            'X': pauli_prob['X'] * T,
            'Y': pauli_prob['Y'] * T,
            'Z': pauli_prob['Z'] * T,
            'None': 1.0 - (pauli_prob['X'] + pauli_prob['Y'] + pauli_prob['Z']) * T
        }
        
        # Tạo mặt nạ ngẫu nhiên để chọn cổng Pauli
        batch_size, channels, height, width = features_float.shape
        pauli_choices = ['X', 'Y', 'Z', 'None']
        probabilities = [scaled_prob['X'], scaled_prob['Y'], scaled_prob['Z'], scaled_prob['None']]
        choice_tensor = torch.multinomial(
            torch.tensor(probabilities, device=device),
            batch_size * channels * height * width,
            replacement=True
        ).view(batch_size, channels, height, width)
        
        # Khởi tạo tensor đầu ra
        noisy_features = features_float.clone()
        
        # Áp dụng cổng Pauli
        for i, pauli in enumerate(pauli_choices):
            mask = (choice_tensor == i)
            if pauli == 'X':
                # Cổng Pauli X: Lật giá trị pixel (giả sử giá trị đã chuẩn hóa trong [0, 1])
                noisy_features[mask] = 1.0 - noisy_features[mask]
            elif pauli == 'Y':
                # Cổng Pauli Y: Kết hợp lật bit và thêm nhiễu ngẫu nhiên
                noisy_features[mask] = 1.0 - noisy_features[mask] + 0.1 * torch.randn_like(noisy_features[mask], device=device)
            elif pauli == 'Z':
                # Cổng Pauli Z: Đổi dấu giá trị pixel
                noisy_features[mask] = -noisy_features[mask]
            # 'None': Giữ nguyên giá trị
            
        # Đảm bảo giá trị pixel nằm trong phạm vi [0, 1]
        noisy_features = torch.clamp(noisy_features, 0.0, 1.0)
        
        return noisy_features
    
    except RuntimeError as e:
        print(f"Quantum noise injection failed: {e}. Returning original features.")
        return features_float