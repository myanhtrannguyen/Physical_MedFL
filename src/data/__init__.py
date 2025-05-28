"""
Unified data handling modules for federated learning with medical imaging datasets.
Supports ACDC (cardiac segmentation) and BraTS2020 (brain tumor segmentation).
"""

# Unified Dataset System
from .dataset import (
    BaseUnifiedDataset,
    ACDCUnifiedDataset,
    BraTS2020UnifiedDataset,
    UnifiedDatasetManager,
    create_unified_dataset,
    create_multi_dataset_loader,
    DatasetType
)

# Unified Loader System
from .loader import (
    UnifiedFederatedLoader,
    create_acdc_loader,
    create_brats_loader,
    create_multi_medical_loader
)

# Core Components
from .preprocessing import (
    MedicalImagePreprocessor,
    DataAugmentation
)

from .partitioning import (
    FederatedDataPartitioner,
    create_federated_splits
)

__all__ = [
    # Unified Dataset Classes
    'BaseUnifiedDataset',
    'ACDCUnifiedDataset',
    'BraTS2020UnifiedDataset',
    'UnifiedDatasetManager',
    'create_unified_dataset',
    'create_multi_dataset_loader',
    'DatasetType',
    
    # Unified Loader Classes
    'UnifiedFederatedLoader',
    'create_acdc_loader',
    'create_brats_loader',
    'create_multi_medical_loader',
    
    # Core Components
    'MedicalImagePreprocessor',
    'DataAugmentation',
    'FederatedDataPartitioner',
    'create_federated_splits'
]
