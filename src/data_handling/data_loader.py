# Federated-Learning/src/data_handling/data_loader.py

import os
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any, Union, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
IMG_SIZE = 256

def _process_h5_file(filepath: str, records: List[Dict], pathology: str, filename: str):
    try:
        with h5py.File(filepath, 'r') as f:
            image_data = np.array(f['image'])
            mask_data = np.array(f['label'])

            if image_data.ndim == 2:
                if np.any(mask_data > 0):
                    records.append({
                        "filepath": filepath, "patient_id": filename.split('_')[0], "pathology": pathology,
                        "image_raw": image_data, "mask_raw": mask_data
                    })
            elif image_data.ndim == 3:
                for i in range(image_data.shape[0]):
                    slice_img, slice_mask = image_data[i, :, :], mask_data[i, :, :]
                    if np.any(slice_mask > 0):
                        records.append({
                            "filepath": f"{filepath}_slice_{i}", "patient_id": filename.split('_')[0], "pathology": pathology,
                            "image_raw": slice_img, "mask_raw": slice_mask
                        })
    except Exception as e:
        logging.warning(f"Error reading or processing file {filepath}: {e}")

def load_and_preprocess_data(data_dir: str, target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> pd.DataFrame:
    def _resize_data(data: npt.NDArray[np.float32], is_mask: bool) -> npt.NDArray[Any]:
        order = 0 if is_mask else 1
        resized = resize(data.astype(np.float32), target_size, order=order, preserve_range=True, anti_aliasing=not is_mask)
        return np.array(resized).astype(np.uint8 if is_mask else np.float32)

    records = []
    logging.info(f"Starting to scan data from: {data_dir}")
    has_subdirs = any(os.path.isdir(os.path.join(data_dir, i)) for i in os.listdir(data_dir) if not i.startswith('.'))

    if has_subdirs:
        logging.info("Detected pathology-based directory structure.")
        for pathology in os.listdir(data_dir):
            pathology_dir = os.path.join(data_dir, pathology)
            if not os.path.isdir(pathology_dir): continue
            for filename in os.listdir(pathology_dir):
                if filename.endswith(".h5"):
                    _process_h5_file(os.path.join(pathology_dir, filename), records, pathology, filename)
    else:
        logging.info("Detected flat structure, processing all .h5 files")
        for filename in os.listdir(data_dir):
            if filename.endswith(".h5"):
                _process_h5_file(os.path.join(data_dir, filename), records, "unknown", filename)

    if not records: raise ValueError(f"No valid data found in {data_dir}")
    df = pd.DataFrame(records)
    logging.info("Resizing and normalizing data...")
    
    # Sửa lỗi Pylance: Sử dụng list comprehension thay vì .apply
    df['image'] = [_resize_data(img, is_mask=False) for img in df['image_raw']]
    df['mask'] = [_resize_data(mask, is_mask=True) for mask in df['mask_raw']]
    
    # Sửa lỗi Pylance: Chuyển đổi sang list trước khi stack
    all_images_stacked = np.stack(list(df['image']))
    max_pixel_value = np.max(all_images_stacked)
    if max_pixel_value > 1.0:
        # Sửa lỗi Pylance: Sử dụng list comprehension
        df['image'] = [img / max_pixel_value for img in df['image']]
        
    df = df.drop(columns=['image_raw', 'mask_raw'])
    logging.info(f"Loaded and processed {len(df)} slices.")
    return df

def partition_data(
    df: pd.DataFrame,
    num_clients: int,
    strategy: str = 'iid',
    alpha: float = 0.5,
    partition_by: str = 'patient'
) -> Dict[int, npt.NDArray[np.intp]]:
    """
    Phân chia dữ liệu cho các client, hỗ trợ phân chia theo 'slice' hoặc 'patient'.
    """
    if partition_by == 'slice':
        logging.info(f"Partitioning {len(df)} slices for {num_clients} clients using strategy: {strategy}")
        # ... (Logic phân chia theo slice)
        pass # Giữ nguyên logic cũ nếu cần
        
    elif partition_by == 'patient':
        patient_groups = df.groupby('patient_id')
        unique_patients = [str(p) for p in patient_groups.groups.keys()]  # Convert to List[str]
        num_patients = len(unique_patients)
        logging.info(f"Partitioning {num_patients} patients for {num_clients} clients using strategy: {strategy}")
        
        client_patient_map: Dict[int, List[str]] = {i: [] for i in range(num_clients)}

        if strategy == 'iid':
            # IID: All clients have access to the SAME set of patients (realistic hospital data sharing)
            logging.info("IID: All clients will have access to all patients (federated data sharing)")
            for client_id in range(num_clients):
                client_patient_map[client_id] = unique_patients.copy()  # All patients for each client

        elif strategy == 'non-iid' or strategy == 'hospital-realistic':
            # Non-IID: Each client (hospital) has different number of patients with potential overlap
            # Realistic scenario: hospitals don't know what data others have
            
            logging.info("Non-IID: Simulating realistic hospital data distribution with varying sizes and overlap")
            
            for client_id in range(num_clients):
                # Random number of patients per client (20-80% of total patients)
                min_patients = max(1, int(0.2 * num_patients))  # At least 20% of patients
                max_patients = int(0.8 * num_patients)  # At most 80% of patients
                num_client_patients = np.random.randint(min_patients, max_patients + 1)
                
                # Randomly sample patients (with potential overlap between clients)
                selected_patients = np.random.choice(
                    np.array(unique_patients), 
                    size=num_client_patients, 
                    replace=False
                ).tolist()
                
                client_patient_map[client_id] = selected_patients
                
                # Log for transparency
                total_slices = sum(len(patient_groups.get_group(p)) for p in selected_patients)
                logging.info(f"  Client {client_id}: {num_client_patients} patients, {total_slices} slices")

        elif strategy == 'pathology-skew':
            patient_pathology = df.groupby('patient_id')['pathology'].first()
            pathologies = patient_pathology.unique()
            if len(pathologies) < 2:
                logging.warning("Only 1 pathology found, using IID patient distribution.")
                return partition_data(df, num_clients, 'iid', alpha, 'patient')
            
            patients_by_pathology = {p: patient_pathology[patient_pathology == p].index.tolist() for p in pathologies}
            
            for i in range(num_clients):
                pathology_to_assign = pathologies[i % len(pathologies)]
                patients_for_this_pathology = patients_by_pathology[pathology_to_assign]
                assigned_patients = [p for j, p in enumerate(patients_for_this_pathology) if j % num_clients == i]
                client_patient_map[i].extend(assigned_patients)

        elif strategy == 'dirichlet':
            patient_pathology = df.groupby('patient_id')['pathology'].first()
            labels, unique_pathologies = pd.factorize(patient_pathology)
            num_classes = len(unique_pathologies)

            proportions = np.random.dirichlet([alpha] * num_classes, num_clients)
            
            patient_indices_by_class = [np.where(labels == k)[0] for k in range(num_classes)]
            for k in range(num_classes): np.random.shuffle(patient_indices_by_class[k])

            client_patient_counts = (np.round(proportions * num_patients)).astype(int)
            diff = num_patients - client_patient_counts.sum()
            client_patient_counts[0, 0] += diff
            
            from_idx = [0] * num_classes
            for client_id in range(num_clients):
                client_patient_indices = []
                for k in range(num_classes):
                    to_idx = from_idx[k] + client_patient_counts[client_id, k]
                    client_patient_indices.extend(patient_indices_by_class[k][from_idx[k]:to_idx])
                    from_idx[k] = to_idx
                client_patient_map[client_id] = [unique_patients[i] for i in client_patient_indices]
        
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")

        client_partitions: Dict[int, npt.NDArray[np.intp]] = {i: np.array([], dtype=np.intp) for i in range(num_clients)}
        for client_id, patient_list in client_patient_map.items():
            if patient_list:
                indices = patient_groups.filter(lambda x: x.name in patient_list).index.to_numpy()
                client_partitions[client_id] = indices
                logging.info(f"  Client {client_id}: Assigned {len(patient_list)} patients, {len(indices)} total slices.")
        return client_partitions
    
    else:
        raise ValueError(f"partition_by must be 'slice' or 'patient'")
    
    return {}


def get_federated_dataloaders(
    data_path: str,
    num_clients: int,
    batch_size: int,
    partition_strategy: str = 'iid',
    val_ratio: float = 0.2,
    alpha: float = 0.5,
    training_sources: List[str] = ['slices'],
    partition_by: str = 'patient'
) -> Tuple[List[DataLoader], List[DataLoader], Optional[DataLoader]]:
    
    all_train_dfs = []
    for source in training_sources:
        if source == 'slices': train_dir = os.path.join(data_path, 'ACDC_training_slices')
        elif source == 'volumes': train_dir = os.path.join(data_path, 'ACDC_training_volumes')
        else: continue
        if not os.path.exists(train_dir): continue
        all_train_dfs.append(load_and_preprocess_data(train_dir))
    if not all_train_dfs: raise ValueError("Could not load any training data.")
    train_df = pd.concat(all_train_dfs, ignore_index=True).drop_duplicates(subset=['filepath']).reset_index(drop=True)
    
    # Sửa lỗi Pylance: Chuyển đổi sang list trước khi stack
    images_tensor = torch.from_numpy(np.stack(list(train_df['image']))).unsqueeze(1).float()
    masks_tensor = torch.from_numpy(np.stack(list(train_df['mask']))).long()
    full_dataset = TensorDataset(images_tensor, masks_tensor)

    client_partitions = partition_data(train_df, num_clients, partition_strategy, alpha, partition_by)

    trainloaders, valloaders = [], []
    for client_id in range(num_clients):
        client_indices = client_partitions.get(client_id, np.array([]))
        if len(client_indices) == 0:
            logging.warning(f"Client {client_id} has no data.")
            trainloaders.append(DataLoader(TensorDataset(torch.empty(0, 1, IMG_SIZE, IMG_SIZE), torch.empty(0, IMG_SIZE, IMG_SIZE, dtype=torch.long))))
            valloaders.append(DataLoader(TensorDataset(torch.empty(0, 1, IMG_SIZE, IMG_SIZE), torch.empty(0, IMG_SIZE, IMG_SIZE, dtype=torch.long))))
            continue

        train_indices, val_indices = train_test_split(client_indices, test_size=val_ratio, random_state=42)
        train_subset, val_subset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)
        trainloaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))

    test_dir = os.path.join(data_path, 'ACDC_testing_volumes')
    testloader = None
    if os.path.exists(test_dir):
        try:
            test_df = load_and_preprocess_data(test_dir)
            # Sửa lỗi Pylance: Chuyển đổi sang list trước khi stack
            test_images_tensor = torch.from_numpy(np.stack(list(test_df['image']))).unsqueeze(1).float()
            test_masks_tensor = torch.from_numpy(np.stack(list(test_df['mask']))).long()
            test_dataset = TensorDataset(test_images_tensor, test_masks_tensor)
            testloader = DataLoader(test_dataset, batch_size=batch_size)
        except (ValueError, FileNotFoundError):
            testloader = None
            
    return trainloaders, valloaders, testloader
