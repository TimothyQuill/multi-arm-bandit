"""
Utility Functions for Book Recommendation System

Contains helper functions for data processing, encoding, and feature engineering.
"""

from typing import List, Any, Dict
import numpy as np
from categories import get_categories, get_default_value, is_valid_category_value


def encode_categorical_feature(value: str, category_name: str, unknown_value: str = None) -> List[float]:
    """
    Encode categorical features using one-hot encoding with standardised categories.
    
    Args:
        value: The categorical value to encode
        category_name: Name of the category (must exist in categories.py)
        unknown_value: Value to use for unknown/missing data (defaults to category default)
        
    Returns:
        One-hot encoded feature vector
        
    Example:
        >>> encode_categorical_feature('mobile', 'device')
        [1.0, 0.0, 0.0]  # mobile, tablet, desktop
        
        >>> encode_categorical_feature('unknown_device', 'device')
        [0.0, 0.0, 1.0]  # defaults to 'desktop'
    """
    # Get the standardised categories for this feature
    try:
        categories = get_categories(category_name)
    except KeyError:
        raise ValueError(f"Unknown category name: {category_name}")
    
    # Get default value if not provided
    if unknown_value is None:
        unknown_value = get_default_value(category_name)
    
    # Use default value if the provided value is not valid
    if not is_valid_category_value(category_name, value):
        value = unknown_value
    
    # Create one-hot encoded vector
    features = [0.0] * len(categories)
    if value in categories:
        features[categories.index(value)] = 1.0
    
    return features


def encode_library_branch(branch: str) -> List[float]:
    """
    Encode library branch into feature vector.
    
    Args:
        branch: Library branch identifier (e.g., 'branch_a', 'main_library')
        
    Returns:
        One-hot encoded feature vector for library branch
    """
    return encode_categorical_feature(branch, 'library_branch')


def encode_multiple_categorical_features(feature_dict: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Encode multiple categorical features at once.
    
    Args:
        feature_dict: Dictionary with feature names as keys and values to encode
        
    Returns:
        Dictionary with feature names as keys and encoded vectors as values
        
    Example:
        >>> features = {
        ...     'gender': 'female',
        ...     'device': 'mobile',
        ...     'library_branch': 'branch_a'
        ... }
        >>> encode_multiple_categorical_features(features)
        {
            'gender': [0.0, 1.0, 0.0, 0.0],
            'device': [1.0, 0.0, 0.0],
            'library_branch': [1.0, 0.0, 0.0, ...]
        }
    """
    encoded_features = {}
    
    for feature_name, value in feature_dict.items():
        try:
            # Convert value to string if it's not already
            str_value = str(value) if value is not None else 'unknown'
            encoded_features[feature_name] = encode_categorical_feature(str_value, feature_name)
        except ValueError as e:
            # Skip features that don't have standardised categories
            print(f"Warning: Could not encode feature '{feature_name}': {e}")
            continue
    
    return encoded_features


def flatten_categorical_features(encoded_features: Dict[str, List[float]]) -> List[float]:
    """
    Flatten multiple encoded categorical features into a single vector.
    
    Args:
        encoded_features: Dictionary of encoded feature vectors
        
    Returns:
        Flattened feature vector
        
    Example:
        >>> encoded = {
        ...     'gender': [0.0, 1.0, 0.0],
        ...     'device': [1.0, 0.0, 0.0]
        ... }
        >>> flatten_categorical_features(encoded)
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    """
    flattened = []
    
    # Sort by key for consistent ordering
    for feature_name in sorted(encoded_features.keys()):
        flattened.extend(encoded_features[feature_name])
    
    return flattened


def get_categorical_feature_dimensions() -> Dict[str, int]:
    """
    Get the dimensions (number of categories) for each categorical feature type.
    
    Returns:
        Dictionary mapping feature names to their dimensions
        
    Example:
        >>> get_categorical_feature_dimensions()
        {'gender': 4, 'device': 3, 'library_branch': 18, ...}
    """
    from categories import CATEGORY_MAPPINGS
    
    return {name: len(categories) for name, categories in CATEGORY_MAPPINGS.items()}


def calculate_total_categorical_dimensions(feature_names: List[str]) -> int:
    """
    Calculate total dimensions needed for a list of categorical features.
    
    Args:
        feature_names: List of categorical feature names to include
        
    Returns:
        Total number of dimensions needed
        
    Example:
        >>> calculate_total_categorical_dimensions(['gender', 'device'])
        7  # 4 for gender + 3 for device
    """
    dimensions = get_categorical_feature_dimensions()
    total = 0
    
    for feature_name in feature_names:
        if feature_name in dimensions:
            total += dimensions[feature_name]
        else:
            print(f"Warning: Unknown categorical feature '{feature_name}'")
    
    return total


def normalise_numeric_feature(value: float, min_val: float, max_val: float, 
                             default_val: float = 0.0) -> float:
    """
    Normalise a numeric feature to [0, 1] range.
    
    Args:
        value: Value to normalise
        min_val: Minimum expected value
        max_val: Maximum expected value
        default_val: Default value if input is None or invalid
        
    Returns:
        Normalised value between 0 and 1
    """
    if value is None or not isinstance(value, (int, float)):
        value = default_val
    
    if max_val <= min_val:
        return 0.0
    
    # Clamp value to range
    value = max(min_val, min(max_val, value))
    
    # Normalise to [0, 1]
    return (value - min_val) / (max_val - min_val)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def create_feature_vector(numeric_features: List[float], 
                         categorical_features: Dict[str, str],
                         target_dim: int = None) -> np.ndarray:
    """
    Create a complete feature vector from numeric and categorical features.
    
    Args:
        numeric_features: List of numeric feature values
        categorical_features: Dictionary of categorical feature values
        target_dim: Target dimension for the final vector (pads/truncates if needed)
        
    Returns:
        Combined feature vector as numpy array
    """
    # Encode categorical features
    encoded_categorical = encode_multiple_categorical_features(categorical_features)
    
    # Flatten categorical features
    categorical_vector = flatten_categorical_features(encoded_categorical)
    
    # Combine numeric and categorical features
    combined_features = list(numeric_features) + categorical_vector
    
    # Convert to numpy array
    feature_vector = np.array(combined_features, dtype=np.float32)
    
    # Pad or truncate to target dimension if specified
    if target_dim is not None:
        if len(feature_vector) < target_dim:
            # Pad with zeros
            feature_vector = np.pad(feature_vector, (0, target_dim - len(feature_vector)))
        elif len(feature_vector) > target_dim:
            # Truncate
            feature_vector = feature_vector[:target_dim]
    
    return feature_vector
