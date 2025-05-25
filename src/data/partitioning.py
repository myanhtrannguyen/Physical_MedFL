"""
Data partitioning utilities for federated learning.
Handles IID and non-IID data splits across clients.
"""

import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from collections import defaultdict, Counter
import random

logger = logging.getLogger(__name__)


class FederatedDataPartitioner:
    """Partitioner for creating federated learning data splits."""
    
    def __init__(
        self,
        num_clients: int,
        data_dir: str,
        output_dir: str,
        seed: int = 42
    ):
        """
        Initialize the data partitioner.
        
        Args:
            num_clients: Number of federated clients
            data_dir: Directory containing the raw data
            output_dir: Directory to save partitioned data
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized partitioner for {num_clients} clients")
    
    def create_iid_partition(
        self,
        data_paths: List[str],
        partition_name: str = "iid"
    ) -> Dict[int, List[str]]:
        """
        Create IID (Independent and Identically Distributed) partition.
        
        Args:
            data_paths: List of data file paths
            partition_name: Name for this partition
            
        Returns:
            Dictionary mapping client_id to list of data paths
        """
        logger.info(f"Creating IID partition with {len(data_paths)} samples")
        
        # Shuffle data paths
        shuffled_paths = data_paths.copy()
        np.random.shuffle(shuffled_paths)
        
        # Split data evenly among clients
        samples_per_client = len(shuffled_paths) // self.num_clients
        remainder = len(shuffled_paths) % self.num_clients
        
        client_data = {}
        start_idx = 0
        
        for client_id in range(self.num_clients):
            # Some clients get one extra sample if there's a remainder
            client_samples = samples_per_client + (1 if client_id < remainder else 0)
            end_idx = start_idx + client_samples
            
            client_data[client_id] = shuffled_paths[start_idx:end_idx]
            start_idx = end_idx
            
            logger.info(f"Client {client_id}: {len(client_data[client_id])} samples")
        
        # Save partition info
        self._save_partition_info(client_data, partition_name)
        
        return client_data
    
    def create_non_iid_partition(
        self,
        data_paths: List[str],
        labels: Optional[List[int]] = None,
        alpha: float = 0.5,
        partition_name: str = "non_iid"
    ) -> Dict[int, List[str]]:
        """
        Create non-IID partition using Dirichlet distribution.
        
        Args:
            data_paths: List of data file paths
            labels: List of labels corresponding to data_paths
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            partition_name: Name for this partition
            
        Returns:
            Dictionary mapping client_id to list of data paths
        """
        logger.info(f"Creating non-IID partition with alpha={alpha}")
        
        if labels is None:
            # Extract labels from file paths (assuming patient-based structure)
            labels = self._extract_labels_from_paths(data_paths)
        
        # Group data by labels
        label_to_paths = defaultdict(list)
        for path, label in zip(data_paths, labels):
            label_to_paths[label].append(path)
        
        unique_labels = list(label_to_paths.keys())
        num_classes = len(unique_labels)
        
        logger.info(f"Found {num_classes} unique labels")
        
        # Generate Dirichlet distribution for each class
        client_data = {i: [] for i in range(self.num_clients)}
        
        for label in unique_labels:
            paths_for_label = label_to_paths[label]
            np.random.shuffle(paths_for_label)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            
            # Distribute samples according to proportions
            start_idx = 0
            for client_id in range(self.num_clients):
                num_samples = int(len(paths_for_label) * proportions[client_id])
                if client_id == self.num_clients - 1:  # Last client gets remaining
                    num_samples = len(paths_for_label) - start_idx
                
                end_idx = start_idx + num_samples
                client_data[client_id].extend(paths_for_label[start_idx:end_idx])
                start_idx = end_idx
        
        # Shuffle each client's data
        for client_id in range(self.num_clients):
            np.random.shuffle(client_data[client_id])
            logger.info(f"Client {client_id}: {len(client_data[client_id])} samples")
        
        # Log distribution statistics
        self._log_distribution_stats(client_data, labels, data_paths)
        
        # Save partition info
        self._save_partition_info(client_data, partition_name)
        
        return client_data
    
    def create_pathological_partition(
        self,
        data_paths: List[str],
        labels: Optional[List[int]] = None,
        shards_per_client: int = 2,
        partition_name: str = "pathological"
    ) -> Dict[int, List[str]]:
        """
        Create pathological non-IID partition (each client has only few classes).
        
        Args:
            data_paths: List of data file paths
            labels: List of labels corresponding to data_paths
            shards_per_client: Number of shards (classes) per client
            partition_name: Name for this partition
            
        Returns:
            Dictionary mapping client_id to list of data paths
        """
        logger.info(f"Creating pathological partition with {shards_per_client} shards per client")
        
        if labels is None:
            labels = self._extract_labels_from_paths(data_paths)
        
        # Group data by labels
        label_to_paths = defaultdict(list)
        for path, label in zip(data_paths, labels):
            label_to_paths[label].append(path)
        
        unique_labels = list(label_to_paths.keys())
        num_classes = len(unique_labels)
        
        # Create shards (each shard contains data from one class)
        shards = []
        for label in unique_labels:
            paths = label_to_paths[label]
            np.random.shuffle(paths)
            
            # Split class data into multiple shards
            shard_size = len(paths) // 2  # Each class creates 2 shards
            if shard_size == 0:
                shard_size = 1
            
            for i in range(0, len(paths), shard_size):
                shard = paths[i:i + shard_size]
                if len(shard) > 0:
                    shards.append((label, shard))
        
        # Shuffle shards
        np.random.shuffle(shards)
        
        # Assign shards to clients
        client_data = {i: [] for i in range(self.num_clients)}
        shard_idx = 0
        
        for client_id in range(self.num_clients):
            for _ in range(shards_per_client):
                if shard_idx < len(shards):
                    label, shard_paths = shards[shard_idx]
                    client_data[client_id].extend(shard_paths)
                    shard_idx += 1
        
        # Distribute remaining shards
        while shard_idx < len(shards):
            client_id = shard_idx % self.num_clients
            label, shard_paths = shards[shard_idx]
            client_data[client_id].extend(shard_paths)
            shard_idx += 1
        
        # Shuffle each client's data
        for client_id in range(self.num_clients):
            np.random.shuffle(client_data[client_id])
            logger.info(f"Client {client_id}: {len(client_data[client_id])} samples")
        
        # Log distribution statistics
        self._log_distribution_stats(client_data, labels, data_paths)
        
        # Save partition info
        self._save_partition_info(client_data, partition_name)
        
        return client_data
    
    def _extract_labels_from_paths(self, data_paths: List[str]) -> List[int]:
        """Extract labels from file paths (patient-based)."""
        labels = []
        for path in data_paths:
            # Extract patient ID from path (e.g., patient001 -> 1)
            path_str = str(path)
            if 'patient' in path_str:
                # Find patient number
                import re
                match = re.search(r'patient(\d+)', path_str)
                if match:
                    patient_id = int(match.group(1))
                    # Group patients into classes (e.g., every 10 patients = 1 class)
                    label = patient_id // 10
                    labels.append(label)
                else:
                    labels.append(0)  # Default label
            else:
                labels.append(0)  # Default label
        
        return labels
    
    def _log_distribution_stats(
        self, 
        client_data: Dict[int, List[str]], 
        labels: List[int], 
        data_paths: List[str]
    ):
        """Log statistics about data distribution."""
        # Create path to label mapping
        path_to_label = {path: label for path, label in zip(data_paths, labels)}
        
        # Count labels per client
        for client_id in range(self.num_clients):
            client_labels = [path_to_label[path] for path in client_data[client_id]]
            label_counts = Counter(client_labels)
            logger.info(f"Client {client_id} label distribution: {dict(label_counts)}")
    
    def _save_partition_info(
        self, 
        client_data: Dict[int, List[str]], 
        partition_name: str
    ):
        """Save partition information to JSON file."""
        partition_info = {
            "partition_name": partition_name,
            "num_clients": self.num_clients,
            "seed": self.seed,
            "client_data": {str(k): v for k, v in client_data.items()}
        }
        
        output_file = self.output_dir / f"{partition_name}_partition.json"
        with open(output_file, 'w') as f:
            json.dump(partition_info, f, indent=2)
        
        logger.info(f"Saved partition info to {output_file}")
    
    def load_partition(self, partition_file: str) -> Dict[int, List[str]]:
        """Load partition from JSON file."""
        with open(partition_file, 'r') as f:
            partition_info = json.load(f)
        
        client_data = {
            int(k): v for k, v in partition_info["client_data"].items()
        }
        
        logger.info(f"Loaded partition from {partition_file}")
        return client_data


def create_federated_splits(
    data_dir: str,
    output_dir: str,
    num_clients: int = 5,
    partition_types: List[str] = ["iid", "non_iid"],
    alpha: float = 0.5,
    seed: int = 42
) -> Dict[str, Dict[int, List[str]]]:
    """
    Create multiple federated data partitions.
    
    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save partitions
        num_clients: Number of federated clients
        partition_types: Types of partitions to create
        alpha: Dirichlet concentration parameter for non-IID
        seed: Random seed
        
    Returns:
        Dictionary mapping partition type to client data
    """
    # Collect all data paths
    data_paths = []
    data_dir_path = Path(data_dir)
    
    # Look for medical image files
    for ext in ['*.nii', '*.nii.gz', '*.dcm', '*.h5']:
        data_paths.extend(list(data_dir_path.rglob(ext)))
    
    # Convert to strings and filter out ground truth files
    data_paths = [
        str(p) for p in data_paths 
        if 'gt' not in str(p).lower() and '_gt' not in str(p).lower()
    ]
    
    logger.info(f"Found {len(data_paths)} data files")
    
    if len(data_paths) == 0:
        raise ValueError(f"No data files found in {data_dir}")
    
    # Initialize partitioner
    partitioner = FederatedDataPartitioner(
        num_clients=num_clients,
        data_dir=data_dir,
        output_dir=output_dir,
        seed=seed
    )
    
    # Create partitions
    partitions = {}
    
    for partition_type in partition_types:
        if partition_type == "iid":
            partitions[partition_type] = partitioner.create_iid_partition(
                data_paths, partition_type
            )
        elif partition_type == "non_iid":
            partitions[partition_type] = partitioner.create_non_iid_partition(
                data_paths, alpha=alpha, partition_name=partition_type
            )
        elif partition_type == "pathological":
            partitions[partition_type] = partitioner.create_pathological_partition(
                data_paths, partition_name=partition_type
            )
        else:
            logger.warning(f"Unknown partition type: {partition_type}")
    
    return partitions


def analyze_partition_quality(
    partition_file: str,
    data_dir: str
) -> Dict[str, float]:
    """
    Analyze the quality of a data partition.
    
    Args:
        partition_file: Path to partition JSON file
        data_dir: Directory containing the data
        
    Returns:
        Dictionary with quality metrics
    """
    # Load partition
    with open(partition_file, 'r') as f:
        partition_info = json.load(f)
    
    client_data = {
        int(k): v for k, v in partition_info["client_data"].items()
    }
    
    # Calculate metrics
    client_sizes = [len(paths) for paths in client_data.values()]
    
    metrics = {
        'mean_size': np.mean(client_sizes),
        'std_size': np.std(client_sizes),
        'min_size': np.min(client_sizes),
        'max_size': np.max(client_sizes),
        'size_coefficient_variation': np.std(client_sizes) / np.mean(client_sizes),
        'total_samples': sum(client_sizes)
    }
    
    logger.info(f"Partition quality metrics: {metrics}")
    return metrics 