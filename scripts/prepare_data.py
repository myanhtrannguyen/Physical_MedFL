#!/usr/bin/env python3
"""
Data preparation script for federated learning.
Handles data preprocessing and partitioning for federated clients.
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.partitioning import create_federated_splits, analyze_partition_quality
from data.preprocessing import MedicalImagePreprocessor, compute_intensity_statistics
from utils.seed import set_seed
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_data(config: dict):
    """
    Preprocess raw data according to configuration.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting data preprocessing...")
    
    data_config = config['data']
    preprocessing_config = data_config['preprocessing']
    
    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor(
        target_size=tuple(preprocessing_config['target_size']),
        normalize=preprocessing_config['normalize'],
        clip_percentiles=tuple(preprocessing_config['clip_percentiles']),
        apply_clahe=preprocessing_config['apply_clahe']
    )
    
    # Create processed data directory
    processed_dir = Path(data_config['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preprocessor configured with target size: {preprocessing_config['target_size']}")
    logger.info("Data preprocessing completed")


def create_partitions(config: dict):
    """
    Create federated data partitions.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Creating federated data partitions...")
    
    data_config = config['data']
    federated_config = config['federated']
    
    # Create partitions
    partitions = create_federated_splits(
        data_dir=data_config['data_dir'],
        output_dir=data_config['partitions_dir'],
        num_clients=federated_config['num_clients'],
        partition_types=[federated_config['partition_type']],
        alpha=federated_config.get('alpha', 0.5),
        seed=config['experiment']['seed']
    )
    
    logger.info(f"Created {federated_config['partition_type']} partitions for {federated_config['num_clients']} clients")
    
    # Analyze partition quality
    partition_file = Path(data_config['partitions_dir']) / f"{federated_config['partition_type']}_partition.json"
    if partition_file.exists():
        metrics = analyze_partition_quality(
            str(partition_file),
            data_config['data_dir']
        )
        logger.info("Partition quality metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")


def validate_data_structure(config: dict):
    """
    Validate that the data directory structure is correct.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Validating data structure...")
    
    data_dir = Path(config['data']['data_dir'])
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check for ACDC structure
    if config['data']['dataset'] == 'acdc':
        # Look for patient directories
        patient_dirs = list(data_dir.glob('patient*'))
        if not patient_dirs:
            raise ValueError(f"No patient directories found in {data_dir}")
        
        logger.info(f"Found {len(patient_dirs)} patient directories")
        
        # Check for image and mask files
        total_images = 0
        total_masks = 0
        
        for patient_dir in patient_dirs[:5]:  # Check first 5 patients
            images = list(patient_dir.glob('*.nii'))
            masks = list(patient_dir.glob('*_gt.nii'))
            
            total_images += len([f for f in images if '_gt' not in f.name])
            total_masks += len(masks)
        
        logger.info(f"Sample validation - Images: {total_images}, Masks: {total_masks}")
    
    logger.info("Data structure validation completed")


def main():
    """Main function for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare data for federated learning")
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/experiment/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Run data preprocessing"
    )
    parser.add_argument(
        "--partition", 
        action="store_true",
        help="Create federated partitions"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate data structure"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all preparation steps"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logger(
        name="data_preparation",
        level=log_level,
        log_file="logs/data_preparation.log",
        console_output=True
    )
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Set seed for reproducibility
        set_seed(config['experiment']['seed'])
        
        # Run requested steps
        if args.all or args.validate:
            validate_data_structure(config)
        
        if args.all or args.preprocess:
            preprocess_data(config)
        
        if args.all or args.partition:
            create_partitions(config)
        
        if not any([args.preprocess, args.partition, args.validate, args.all]):
            logger.warning("No preparation steps specified. Use --help for options.")
            return
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main() 