"""
Streamlined data handling for AI research with medical imaging datasets.
Optimized for accuracy and minimal redundancy.
"""

# Core Dataset Classes (Direct import - no wrapper layers)
from .dataset import (
    ACDCUnifiedDataset,
    BraTS2020UnifiedDataset,
    create_unified_dataset,
    DatasetType
)

# Core Preprocessing (Single source of truth)
from .preprocessing import (
    MedicalImagePreprocessor,
    DataAugmentation
)

# Research-focused loaders (Simplified)
from .research_loader import (
    create_research_dataloader,
    create_federated_research_loaders
)

__all__ = [
    # Core Datasets (No manager wrapper)
    'ACDCUnifiedDataset',
    'BraTS2020UnifiedDataset',
    'create_unified_dataset',
    'DatasetType',
    
    # Core Preprocessing (Single path)
    'MedicalImagePreprocessor',
    'DataAugmentation',
    
    # Research Loaders (Optimized)
    'create_research_dataloader',
    'create_federated_research_loaders'
]
