# Federated-Learning/src/data_handling/data_loader.py

import os
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any, Union, Optional
from collections import defaultdict
import logging
import pytorch_lightning as pl

# Lightning imports
try:
    import pytorch_lightning as pl
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not available for DataModule.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
IMG_SIZE = 256


class H5Dataset(Dataset):
    """
    Custom Dataset với lazy loading - chỉ đọc dữ liệu khi cần thiết.
    Tiết kiệm bộ nhớ bằng cách không tải toàn bộ dataset vào memory.
    """
    
    def __init__(self, file_info: pd.DataFrame, target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)):
        """
        Args:
            file_info: DataFrame chứa metadata về các file (filepath, patient_id, slice_idx, etc.)
            target_size: Kích thước target cho resize
        """
        self.file_info = file_info.reset_index(drop=True)
        self.target_size = target_size
        
        # Ước tính max pixel value bằng cách sampling một số file
        self.max_pixel_value = self._estimate_max_pixel_value()
        
        logging.info(f"H5Dataset initialized with {len(self.file_info)} samples")
        logging.info(f"Estimated max pixel value: {self.max_pixel_value}")
    
    def __len__(self) -> int:
        return len(self.file_info)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lazy loading: Chỉ đọc và xử lý dữ liệu khi được yêu cầu
        """
        row = self.file_info.iloc[idx]
        filepath = row['filepath']
        slice_idx = row.get('slice_idx', None)
        
        try:
            with h5py.File(filepath, 'r') as f:
                image_data = np.array(f['image'])
                mask_data = np.array(f['label'])
                
                # Xử lý 2D vs 3D data
                if image_data.ndim == 2:
                    image_slice = image_data
                    mask_slice = mask_data
                elif image_data.ndim == 3:
                    if slice_idx is None:
                        raise ValueError(f"3D data requires slice_idx, but got None for {filepath}")
                    image_slice = image_data[slice_idx]
                    mask_slice = mask_data[slice_idx]
                else:
                    raise ValueError(f"Unsupported image dimensions: {image_data.ndim}")
                
                # Resize và chuẩn hóa
                image_processed = self._resize_and_normalize(image_slice, is_mask=False)
                mask_processed = self._resize_and_normalize(mask_slice, is_mask=True)
                
                # Convert to tensor
                image_tensor = torch.from_numpy(image_processed).unsqueeze(0).float()  # Add channel dim
                mask_tensor = torch.from_numpy(mask_processed).long()
                
                return image_tensor, mask_tensor
                
        except Exception as e:
            logging.error(f"Error loading data from {filepath}, slice {slice_idx}: {e}")
            # Return zero tensors as fallback
            zero_image = torch.zeros(1, *self.target_size, dtype=torch.float32)
            zero_mask = torch.zeros(*self.target_size, dtype=torch.long)
            return zero_image, zero_mask
    
    def _resize_and_normalize(self, data: np.ndarray, is_mask: bool) -> np.ndarray:
        """Resize và chuẩn hóa dữ liệu"""
        # Ensure input is numpy array
        data = np.asarray(data, dtype=np.float32)
        
        # Resize
        order = 0 if is_mask else 1  # Nearest neighbor for masks, bilinear for images
        resized = resize(
            data, 
            self.target_size, 
            order=order, 
            preserve_range=True, 
            anti_aliasing=not is_mask
        )
        
        # Ensure resized is numpy array and handle type conversion properly
        resized = np.asarray(resized)
        
        if is_mask:
            return resized.astype(np.uint8)
        else:
            # Normalize image - fix type casting
            resized_float = resized.astype(np.float32)
            normalized = resized_float / self.max_pixel_value if self.max_pixel_value > 1.0 else resized_float
            return normalized
    
    def _estimate_max_pixel_value(self, sample_size: int = 10) -> float:
        """
        Ước tính max pixel value bằng cách sampling một số file
        """
        max_val = 1.0
        sample_indices = np.random.choice(len(self.file_info), 
                                        size=min(sample_size, len(self.file_info)), 
                                        replace=False)
        
        for idx in sample_indices:
            filepath = "unknown"  # Initialize default value
            try:
                row = self.file_info.iloc[idx]
                filepath = row['filepath']
                slice_idx = row.get('slice_idx', None)
                
                with h5py.File(filepath, 'r') as f:
                    image_data = np.array(f['image'])
                    
                    if image_data.ndim == 2:
                        sample_image = image_data
                    elif image_data.ndim == 3:
                        # Lấy slice giữa nếu không có slice_idx
                        sample_slice_idx = slice_idx if slice_idx is not None else image_data.shape[0] // 2
                        sample_image = image_data[sample_slice_idx]
                    else:
                        continue
                    
                    max_val = max(max_val, np.max(sample_image))
                    
            except Exception as e:
                logging.warning(f"Error sampling file {filepath}: {e}")
                continue
        
        return max_val


def scan_data_directory(data_dir: str) -> pd.DataFrame:
    """
    Quét thư mục dữ liệu và thu thập metadata thay vì tải toàn bộ dữ liệu.
    Trả về DataFrame chứa thông tin về tất cả các slice có thể sử dụng.
    """
    records = []
    logging.info(f"Scanning data directory: {data_dir}")
    
    # Kiểm tra cấu trúc thư mục
    has_subdirs = any(os.path.isdir(os.path.join(data_dir, i)) 
                     for i in os.listdir(data_dir) if not i.startswith('.'))
    
    if has_subdirs:
        logging.info("Detected pathology-based directory structure.")
        for pathology in os.listdir(data_dir):
            pathology_dir = os.path.join(data_dir, pathology)
            if not os.path.isdir(pathology_dir):
                continue
                
            for filename in os.listdir(pathology_dir):
                if filename.endswith(".h5"):
                    filepath = os.path.join(pathology_dir, filename)
                    _scan_h5_file(filepath, records, pathology, filename)
    else:
        logging.info("Detected flat structure, processing all .h5 files")
        for filename in os.listdir(data_dir):
            if filename.endswith(".h5"):
                filepath = os.path.join(data_dir, filename)
                _scan_h5_file(filepath, records, "unknown", filename)
    
    if not records:
        raise ValueError(f"No valid data found in {data_dir}")
    
    df = pd.DataFrame(records)
    logging.info(f"Scanned {len(df)} valid slices from {len(df['filepath'].unique())} files")
    
    return df


def _scan_h5_file(filepath: str, records: List[Dict], pathology: str, filename: str):
    """
    Quét một file H5 và thu thập metadata (không tải pixel data)
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Convert h5py datasets to numpy arrays to access shape
            image_data = np.array(f['image'])
            mask_data = np.array(f['label'])
            patient_id = filename.split('_')[0]
            
            if len(image_data.shape) == 2:
                # 2D data - single slice
                if np.any(mask_data > 0):  # Check if mask has any positive values
                    records.append({
                        "filepath": filepath,
                        "patient_id": patient_id,
                        "pathology": pathology,
                        "slice_idx": None,  # 2D data doesn't need slice index
                        "has_mask": True
                    })
            elif len(image_data.shape) == 3:
                # 3D data - multiple slices
                for i in range(image_data.shape[0]):
                    mask_slice = mask_data[i, :, :]
                    if np.any(mask_slice > 0):  # Only include slices with masks
                        records.append({
                            "filepath": filepath,
                            "patient_id": patient_id,
                            "pathology": pathology,
                            "slice_idx": i,
                            "has_mask": True
                        })
            else:
                logging.warning(f"Unsupported data shape in {filepath}: {image_data.shape}")
                
    except Exception as e:
        logging.warning(f"Error scanning file {filepath}: {e}")


def partition_data(
    df: pd.DataFrame,
    num_clients: int,
    strategy: str = 'iid',
    alpha: float = 0.5,
    partition_by: str = 'patient'
) -> Dict[int, npt.NDArray[np.intp]]:
    """
    Phân chia metadata cho các client - làm việc với metadata thay vì dữ liệu thực tế.
    Đảm bảo tất cả slices của một bệnh nhân chỉ thuộc về một client.
    
    Args:
        df: DataFrame chứa metadata từ scan_data_directory
        num_clients: Số lượng clients
        strategy: Chiến lược phân chia ('iid', 'non-iid', 'pathology-skew', 'dirichlet')
        alpha: Tham số cho Dirichlet distribution
        partition_by: 'patient' để đảm bảo integrity
    """
    if partition_by != 'patient':
        logging.warning("partition_by should be 'patient' to ensure data integrity. Forcing to 'patient'.")
    
    # Nhóm theo patient_id để đảm bảo không tách rời dữ liệu của cùng một bệnh nhân
    patient_groups = df.groupby('patient_id')
    unique_patients = [str(p) for p in patient_groups.groups.keys()]  # Convert to List[str]
    num_patients = len(unique_patients)
    
    logging.info(f"Partitioning {num_patients} patients ({len(df)} total slices) for {num_clients} clients using strategy: {strategy}")
    
    # Khởi tạo mapping client -> list patients
    client_patient_map: Dict[int, List[str]] = {i: [] for i in range(num_clients)}

    if strategy == 'iid':
        # IID: Tất cả clients có quyền truy cập cùng một tập bệnh nhân
        logging.info("IID: All clients will have access to all patients (federated data sharing)")
        for client_id in range(num_clients):
            client_patient_map[client_id] = unique_patients.copy()

    elif strategy == 'non-iid' or strategy == 'hospital-realistic':
        # Non-IID: Mỗi client (bệnh viện) có số lượng bệnh nhân khác nhau với khả năng overlap
        logging.info("Non-IID: Simulating realistic hospital data distribution with varying sizes and overlap")
        
        for client_id in range(num_clients):
            # Số bệnh nhân ngẫu nhiên cho mỗi client (20-80% tổng số bệnh nhân)
            min_patients = max(1, int(0.2 * num_patients))
            max_patients = int(0.8 * num_patients)
            num_client_patients = np.random.randint(min_patients, max_patients + 1)
            
            # Random sampling bệnh nhân (có thể overlap giữa các clients)
            selected_patients = np.random.choice(
                np.array(unique_patients), 
                size=num_client_patients, 
                replace=False
            ).tolist()
            
            client_patient_map[client_id] = selected_patients
            
            # Log thông tin
            total_slices = sum(len(patient_groups.get_group(p)) for p in selected_patients)
            logging.info(f"  Client {client_id}: {num_client_patients} patients, {total_slices} slices")

    elif strategy == 'pathology-skew':
        # Phân chia theo pathology - mỗi client thiên về một loại bệnh
        patient_pathology = df.groupby('patient_id')['pathology'].first()
        pathologies = patient_pathology.unique()
        
        if len(pathologies) < 2:
            logging.warning("Only 1 pathology found, falling back to IID patient distribution.")
            return partition_data(df, num_clients, 'iid', alpha, 'patient')
        
        logging.info(f"Pathology-skew: Distributing {len(pathologies)} pathologies across {num_clients} clients")
        
        patients_by_pathology = {
            pathology: patient_pathology[patient_pathology == pathology].index.tolist() 
            for pathology in pathologies
        }
        
        # Gán bệnh nhân theo pathology cho từng client
        for i in range(num_clients):
            pathology_to_assign = pathologies[i % len(pathologies)]
            patients_for_this_pathology = patients_by_pathology[pathology_to_assign]
            
            # Chia đều bệnh nhân của pathology này cho các clients
            assigned_patients = [
                p for j, p in enumerate(patients_for_this_pathology) 
                if j % num_clients == i
            ]
            client_patient_map[i].extend(assigned_patients)
            
            total_slices = sum(len(patient_groups.get_group(p)) for p in assigned_patients)
            logging.info(f"  Client {i} ({pathology_to_assign}): {len(assigned_patients)} patients, {total_slices} slices")

    elif strategy == 'dirichlet':
        # Dirichlet distribution cho việc phân chia non-IID
        logging.info(f"Dirichlet: Using alpha={alpha} for non-IID distribution")
        
        patient_pathology = df.groupby('patient_id')['pathology'].first()
        labels, unique_pathologies = pd.factorize(patient_pathology)
        num_classes = len(unique_pathologies)

        # Tạo proportions từ Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_classes, num_clients)
        
        # Group patients theo class
        patient_indices_by_class = [
            np.where(labels == k)[0] for k in range(num_classes)
        ]
        for k in range(num_classes): 
            np.random.shuffle(patient_indices_by_class[k])

        # Tính số lượng bệnh nhân cho mỗi client và class
        client_patient_counts = (np.round(proportions * num_patients)).astype(int)
        
        # Điều chỉnh để tổng bằng num_patients
        diff = num_patients - client_patient_counts.sum()
        client_patient_counts[0, 0] += diff
        
        # Phân chia bệnh nhân
        from_idx = [0] * num_classes
        for client_id in range(num_clients):
            client_patient_indices = []
            for k in range(num_classes):
                to_idx = from_idx[k] + client_patient_counts[client_id, k]
                client_patient_indices.extend(
                    patient_indices_by_class[k][from_idx[k]:to_idx]
                )
                from_idx[k] = to_idx
            
            client_patient_map[client_id] = [unique_patients[i] for i in client_patient_indices]
            
            total_slices = sum(len(patient_groups.get_group(p)) for p in client_patient_map[client_id])
            logging.info(f"  Client {client_id}: {len(client_patient_map[client_id])} patients, {total_slices} slices")
    
    else:
        raise ValueError(f"Unknown partitioning strategy: {strategy}")

    # Chuyển đổi từ patient mapping sang slice indices
    client_partitions: Dict[int, npt.NDArray[np.intp]] = {}
    for client_id, patient_list in client_patient_map.items():
        if patient_list:
            # Lấy tất cả indices của slices thuộc về các bệnh nhân này
            client_indices = []
            for patient_id in patient_list:
                if patient_id in patient_groups.groups:
                    patient_slice_indices = patient_groups.get_group(patient_id).index.tolist()
                    client_indices.extend(patient_slice_indices)
            
            client_partitions[client_id] = np.array(client_indices, dtype=np.intp)
            logging.info(f"  Final - Client {client_id}: {len(patient_list)} patients, {len(client_indices)} slices")
        else:
            client_partitions[client_id] = np.array([], dtype=np.intp)
            logging.warning(f"  Client {client_id}: No patients assigned!")
            
    return client_partitions


def get_federated_dataloaders(
    data_path: str,
    num_clients: int,
    batch_size: int,
    partition_strategy: str = 'iid',
    val_ratio: float = 0.2,
    alpha: float = 0.5,
    training_sources: List[str] = ['slices'],
    partition_by: str = 'patient',
    num_workers: int = 2
) -> Tuple[List[DataLoader], List[DataLoader], Optional[DataLoader]]:
    """
    Tạo federated dataloaders với lazy loading - chỉ tải dữ liệu khi cần thiết.
    
    Args:
        data_path: Đường dẫn đến thư mục dữ liệu chính
        num_clients: Số lượng clients
        batch_size: Batch size cho DataLoader
        partition_strategy: Chiến lược phân chia dữ liệu
        val_ratio: Tỷ lệ validation
        alpha: Tham số cho Dirichlet distribution
        training_sources: Danh sách nguồn training data ['slices', 'volumes']
        partition_by: Phân chia theo 'patient' để đảm bảo integrity
        num_workers: Số workers cho DataLoader (song song hóa)
    
    Returns:
        Tuple[trainloaders, valloaders, testloader]
    """
    
    # Thu thập metadata từ các nguồn training data
    all_train_dfs = []
    for source in training_sources:
        if source == 'slices': 
            train_dir = os.path.join(data_path, 'ACDC_training_slices')
        elif source == 'volumes': 
            train_dir = os.path.join(data_path, 'ACDC_training_volumes')
        else: 
            continue
            
        if not os.path.exists(train_dir): 
            continue
            
        # Sử dụng scan_data_directory thay vì load_and_preprocess_data
        source_df = scan_data_directory(train_dir)
        all_train_dfs.append(source_df)
    
    if not all_train_dfs: 
        raise ValueError("Could not load any training data.")
    
    # Kết hợp metadata từ tất cả các nguồn
    train_df = pd.concat(all_train_dfs, ignore_index=True).drop_duplicates(subset=['filepath']).reset_index(drop=True)
    logging.info(f"Combined training metadata: {len(train_df)} total slices from {len(train_df['filepath'].unique())} files")
    
    # Tạo H5Dataset với lazy loading
    full_dataset = H5Dataset(train_df)
    logging.info(f"Created H5Dataset with {len(full_dataset)} samples")

    # Phân chia metadata cho các clients
    client_partitions = partition_data(train_df, num_clients, partition_strategy, alpha, partition_by)

    # Tạo dataloaders cho từng client
    trainloaders, valloaders = [], []
    for client_id in range(num_clients):
        client_indices = client_partitions.get(client_id, np.array([]))
        
        if len(client_indices) == 0:
            logging.warning(f"Client {client_id} has no data. Creating empty dataloaders.")
            # Tạo empty dataset với đúng format
            empty_df = pd.DataFrame(columns=train_df.columns)
            empty_dataset = H5Dataset(empty_df)
            trainloaders.append(DataLoader(empty_dataset, batch_size=batch_size, num_workers=0))
            valloaders.append(DataLoader(empty_dataset, batch_size=batch_size, num_workers=0))
            continue

        # Chia train/val cho client này
        train_indices, val_indices = train_test_split(
            client_indices, 
            test_size=val_ratio, 
            random_state=42
        )
        
        # Tạo Subset từ H5Dataset
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        # Tạo DataLoaders với num_workers để tận dụng song song hóa
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        trainloaders.append(train_loader)
        valloaders.append(val_loader)
        
        logging.info(f"Client {client_id}: {len(train_indices)} train samples, {len(val_indices)} val samples")

    # Tạo test dataloader nếu có test data
    test_dir = os.path.join(data_path, 'ACDC_testing_volumes')
    testloader = None
    
    if os.path.exists(test_dir):
        try:
            # Quét test data metadata
            test_df = scan_data_directory(test_dir)
            logging.info(f"Found test data: {len(test_df)} slices")
            
            # Tạo H5Dataset cho test data
            test_dataset = H5Dataset(test_df)
            testloader = DataLoader(
                test_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            logging.info(f"Created test dataloader with {len(test_dataset)} samples")
            
        except (ValueError, FileNotFoundError) as e:
            logging.warning(f"Could not create test dataloader: {e}")
            testloader = None
    else:
        logging.info("No test directory found")
            
    return trainloaders, valloaders, testloader


# Lightning DataModule Integration
if LIGHTNING_AVAILABLE:
    class LightningACDCDataModule(pl.LightningDataModule):
        """PyTorch Lightning DataModule for ACDC dataset with federated learning support."""
        
        def __init__(
            self,
            data_path: str,
            num_clients: int = 2,
            batch_size: int = 8,
            num_workers: int = 4,
            client_id: Optional[int] = None,
            split_method: str = 'pathology_aware',
            iid_fraction: float = 0.1,
            val_split: float = 0.2,
            seed: int = 42
        ):
            super().__init__()
            self.save_hyperparameters()
            
            self.data_path = data_path
            self.num_clients = num_clients
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.client_id = client_id
            self.split_method = split_method
            self.iid_fraction = iid_fraction
            self.val_split = val_split
            self.seed = seed
            
            # Datasets will be initialized in setup()
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            self.num_train_examples = 0
            
        def prepare_data(self) -> None:
            """Download or prepare data. Called only once per node."""
            # Data should already be preprocessed and available
            train_dir = os.path.join(self.data_path, 'ACDC_training_volumes')
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"Training data directory not found: {train_dir}")
        
        def setup(self, stage: Optional[str] = None) -> None:
            """Setup datasets for each stage."""
            
            if stage == 'fit' or stage is None:
                # Load training data
                train_dir = os.path.join(self.data_path, 'ACDC_training_volumes')
                train_df = scan_data_directory(train_dir)
                
                if self.client_id is not None:
                    # Federated learning setup - get data for specific client
                    client_partitions = partition_data(
                        train_df,
                        num_clients=self.num_clients,
                        strategy=self.split_method,
                        alpha=0.5,  # Default alpha for non-IID
                        partition_by='patient'
                    )
                    
                    # Get indices for this client
                    client_indices = client_partitions.get(self.client_id, np.array([]))
                    if len(client_indices) == 0:
                        raise ValueError(f"No data assigned to client {self.client_id}")
                    
                    # Get data for this client
                    client_data = train_df.iloc[client_indices].reset_index(drop=True)
                    
                    # Split into train/val
                    train_indices, val_indices = train_test_split(
                        range(len(client_data)),
                        test_size=self.val_split,
                        random_state=self.seed,
                        stratify=client_data['pathology'] if 'pathology' in client_data.columns else None
                    )
                    
                    self.train_dataset = H5Dataset(client_data.iloc[train_indices].reset_index(drop=True))
                    self.val_dataset = H5Dataset(client_data.iloc[val_indices].reset_index(drop=True))
                    self.num_train_examples = len(self.train_dataset)
                    
                else:
                    # Centralized training setup
                    train_indices, val_indices = train_test_split(
                        range(len(train_df)),
                        test_size=self.val_split,
                        random_state=self.seed,
                        stratify=train_df['pathology'] if 'pathology' in train_df.columns else None
                    )
                    
                    self.train_dataset = H5Dataset(train_df.iloc[train_indices].reset_index(drop=True))
                    self.val_dataset = H5Dataset(train_df.iloc[val_indices].reset_index(drop=True))
                    self.num_train_examples = len(self.train_dataset)
            
            if stage == 'test' or stage is None:
                # Load test data
                test_dir = os.path.join(self.data_path, 'ACDC_testing_volumes')
                if os.path.exists(test_dir):
                    test_df = scan_data_directory(test_dir)
                    self.test_dataset = H5Dataset(test_df)
        
        def train_dataloader(self) -> DataLoader:
            """Return training dataloader."""
            if self.train_dataset is None:
                raise ValueError("Training dataset not initialized. Call setup() first.")
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if self.num_workers > 0 else False
            )
        
        def val_dataloader(self) -> DataLoader:
            """Return validation dataloader."""
            if self.val_dataset is None:
                raise ValueError("Validation dataset not initialized. Call setup() first.")
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if self.num_workers > 0 else False
            )
        
        def test_dataloader(self) -> Optional[DataLoader]:
            """Return test dataloader."""
            if self.test_dataset is not None:
                return DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=True if self.num_workers > 0 else False
                )
            return None
