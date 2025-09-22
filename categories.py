"""
Categorical Variables for Book Recommendation System

Contains standardised lists of categorical variables used throughout the system.
"""

# Gender categories
GENDER_CATEGORIES = ['male', 'female', 'other', 'unknown']

# Occupation categories (simplified from education + occupation)
OCCUPATION_CATEGORIES = ['student', 'professional', 'retired', 'unemployed', 'other', 'unknown']

# Time of day categories
TIME_OF_DAY_CATEGORIES = ['morning', 'afternoon', 'evening', 'night']

# Device categories
DEVICE_CATEGORIES = ['mobile', 'tablet', 'desktop']

# Library branch categories
LIBRARY_BRANCH_CATEGORIES = [
    'branch_a', 'branch_b', 'branch_c', 'branch_d', 'branch_e',
    'branch_f', 'branch_g', 'branch_h', 'branch_i', 'branch_j',
    'branch_k', 'branch_l', 'branch_m', 'branch_n', 'branch_o',
    'main_library', 'central_branch', 'unknown'
]

# Book genre categories
BOOK_GENRE_CATEGORIES = [
    'fiction', 'nonfiction', 'mystery', 'romance', 'scifi', 
    'biography', 'history', 'fantasy', 'thriller', 'drama',
    'poetry', 'young_adult', 'children', 'self_help', 'business'
]

# User action categories (for interactions)
ACTION_CATEGORIES = ['view', 'click', 'borrow', 'return', 'favourite', 'ignore']

# Age group categories (derived from numeric age)
AGE_GROUP_CATEGORIES = ['child', 'teen', 'young_adult', 'adult', 'senior', 'unknown']

# Reading frequency categories
READING_FREQUENCY_CATEGORIES = ['rarely', 'monthly', 'weekly', 'daily', 'unknown']

# Book format categories
BOOK_FORMAT_CATEGORIES = ['physical', 'ebook', 'audiobook', 'magazine', 'newspaper']

# User membership tier categories
MEMBERSHIP_TIER_CATEGORIES = ['basic', 'premium', 'student', 'senior', 'family']

# Collection type categories
COLLECTION_TYPE_CATEGORIES = ['general', 'reference', 'rare_books', 'childrens', 'local_history']

# Category mappings for easy access
CATEGORY_MAPPINGS = {
    'gender': GENDER_CATEGORIES,
    'occupation_category': OCCUPATION_CATEGORIES,
    'time_of_day': TIME_OF_DAY_CATEGORIES,
    'device': DEVICE_CATEGORIES,
    'library_branch': LIBRARY_BRANCH_CATEGORIES,
    'genre': BOOK_GENRE_CATEGORIES,
    'action': ACTION_CATEGORIES,
    'age_group': AGE_GROUP_CATEGORIES,
    'reading_frequency': READING_FREQUENCY_CATEGORIES,
    'book_format': BOOK_FORMAT_CATEGORIES,
    'membership_tier': MEMBERSHIP_TIER_CATEGORIES,
    'collection_type': COLLECTION_TYPE_CATEGORIES
}

# Default mappings for categorical features with their default values
DEFAULT_CATEGORICAL_VALUES = {
    'gender': 'unknown',
    'occupation_category': 'unknown',
    'time_of_day': 'afternoon',
    'device': 'desktop',
    'library_branch': 'unknown',
    'genre': 'fiction',
    'action': 'view',
    'age_group': 'unknown',
    'reading_frequency': 'unknown',
    'book_format': 'physical',
    'membership_tier': 'basic',
    'collection_type': 'general'
}

# Helper function to get category list by name
def get_categories(category_name: str):
    """
    Get category list by name.
    
    Args:
        category_name: Name of the category
        
    Returns:
        List of category values
        
    Raises:
        KeyError: If category name not found
    """
    if category_name not in CATEGORY_MAPPINGS:
        raise KeyError(f"Category '{category_name}' not found. Available categories: {list(CATEGORY_MAPPINGS.keys())}")
    
    return CATEGORY_MAPPINGS[category_name]

# Helper function to get default value for a category
def get_default_value(category_name: str):
    """
    Get default value for a category.
    
    Args:
        category_name: Name of the category
        
    Returns:
        Default value for the category
        
    Raises:
        KeyError: If category name not found
    """
    if category_name not in DEFAULT_CATEGORICAL_VALUES:
        raise KeyError(f"Default value for category '{category_name}' not found.")
    
    return DEFAULT_CATEGORICAL_VALUES[category_name]

# Helper function to validate if a value exists in a category
def is_valid_category_value(category_name: str, value: str) -> bool:
    """
    Check if a value is valid for a given category.
    
    Args:
        category_name: Name of the category
        value: Value to check
        
    Returns:
        True if value is valid, False otherwise
    """
    try:
        categories = get_categories(category_name)
        return value in categories
    except KeyError:
        return False
