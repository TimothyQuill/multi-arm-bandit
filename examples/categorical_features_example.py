"""
Example demonstrating how to handle categorical features in the contextual bandit model.

This example shows:
1. How to encode categorical features like gender, postcode, education level
2. How to use the feature encoding methods
3. How to provide user context with categorical data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.contextual_bandit import ContextualBandit, BanditConfig
import numpy as np
from datetime import datetime

def main():
    # Initialize the bandit with appropriate feature dimension
    # We need to account for all categorical features
    config = BanditConfig(
        alpha=1.0,
        feature_dim=100,  # Increased to accommodate categorical features
        regularization=1.0,
        exploration_rate=0.1
    )
    
    bandit = ContextualBandit(config)
    
    # Example user contexts with categorical features
    user_contexts = [
        {
            'user_id': 'user_001',
            'age': 25,
            'gender': 'female',
            'postcode': 'SW1A 1AA',  # London
            'education_level': 'bachelor',
            'occupation_category': 'student',
            'preference_fiction': 0.8,
            'preference_nonfiction': 0.2,
            'preference_mystery': 0.6,
            'preference_romance': 0.9,
            'preference_scifi': 0.3,
            'avg_reading_time': 45,
            'borrow_frequency': 3
        },
        {
            'user_id': 'user_002',
            'age': 45,
            'gender': 'male',
            'postcode': 'M1 1AA',  # Manchester
            'education_level': 'master',
            'occupation_category': 'professional',
            'preference_fiction': 0.3,
            'preference_nonfiction': 0.9,
            'preference_mystery': 0.4,
            'preference_romance': 0.1,
            'preference_scifi': 0.7,
            'avg_reading_time': 60,
            'borrow_frequency': 5
        },
        {
            'user_id': 'user_003',
            'age': 65,
            'gender': 'other',
            'postcode': 'EH1 1AA',  # Edinburgh
            'education_level': 'phd',
            'occupation_category': 'retired',
            'preference_fiction': 0.5,
            'preference_nonfiction': 0.7,
            'preference_mystery': 0.8,
            'preference_romance': 0.2,
            'preference_scifi': 0.4,
            'avg_reading_time': 90,
            'borrow_frequency': 2
        }
    ]
    
    # Example session context
    session_context = {
        'time_of_day': 'afternoon',
        'device': 'desktop',
        'duration_minutes': 30
    }
    
    # Add some example books
    books = ['book_001', 'book_002', 'book_003', 'book_004', 'book_005']
    for book_id in books:
        bandit.add_arm(book_id)
    
    print("=== Categorical Features Example ===\n")
    
    # Test feature extraction for each user
    for i, user_context in enumerate(user_contexts):
        print(f"User {i+1}: {user_context['user_id']}")
        print(f"  Gender: {user_context['gender']}")
        print(f"  Postcode: {user_context['postcode']}")
        print(f"  Education: {user_context['education_level']}")
        print(f"  Occupation: {user_context['occupation_category']}")
        
        # Extract features
        features = bandit.extract_features(user_context, session_context, {})
        print(f"  Feature vector length: {len(features)}")
        print(f"  Non-zero features: {np.count_nonzero(features)}")
        print()
    
    # Test recommendations with categorical features
    print("=== Recommendations with Categorical Features ===\n")
    
    user_context = user_contexts[0]  # Use first user
    recommendations = bandit.select_arm(user_context, session_context, books, n_recommendations=3)
    
    print(f"Recommendations for {user_context['user_id']}:")
    for book_id, confidence in recommendations:
        print(f"  {book_id}: {confidence:.4f}")
    
    # Test interaction recording
    print("\n=== Recording Interactions ===\n")
    
    # Record some interactions
    for i, (book_id, confidence) in enumerate(recommendations):
        action = 'borrow' if i == 0 else 'view'  # First book borrowed, others viewed
        dwell_time = 120 if i == 0 else 30  # More time on borrowed book
        
        bandit.record_interaction(
            user_id=user_context['user_id'],
            book_id=book_id,
            action=action,
            dwell_time=dwell_time
        )
        
        print(f"Recorded: {action} of {book_id} (dwell: {dwell_time}s)")
    
    # Show arm statistics
    print("\n=== Arm Statistics ===\n")
    stats = bandit.get_arm_statistics()
    for book_id, stat in stats.items():
        print(f"{book_id}: count={stat['count']}, theta_norm={stat['theta_norm']:.4f}")

if __name__ == "__main__":
    main() 