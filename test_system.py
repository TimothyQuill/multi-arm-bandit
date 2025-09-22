"""
Simple test script for the Book Recommender System

Tests the core functionality without requiring external dependencies.
"""

import sys
import os
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.contextual_bandit import BookRecommenderBandit, BanditConfig
    print("✓ Successfully imported contextual bandit model")
except ImportError as e:
    print(f"✗ Failed to import contextual bandit model: {e}")
    sys.exit(1)

try:
    from config.settings import get_settings
    print("✓ Successfully imported settings")
except ImportError as e:
    print(f"✗ Failed to import settings: {e}")
    sys.exit(1)


def test_bandit_initialization():
    """Test bandit model initialization."""
    print("\n--- Testing Bandit Initialization ---")
    
    try:
        config = BanditConfig(
            alpha=1.0,
            feature_dim=20,  # Smaller for testing
            regularization=1.0,
            exploration_rate=0.1,
            decay_rate=0.99,
            min_observations=3
        )
        
        bandit = BookRecommenderBandit(config)
        print("✓ Bandit model initialized successfully")
        return bandit
        
    except Exception as e:
        print(f"✗ Failed to initialize bandit: {e}")
        return None


def test_book_management(bandit):
    """Test adding and managing books."""
    print("\n--- Testing Book Management ---")
    
    try:
        # Add sample books
        sample_books = {
            "book_001": {
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "genre": "fiction",
                "popularity_score": 0.9,
                "rating": 4.2,
                "page_count": 180,
                "publication_year": 1925,
                "availability": 1.0
            },
            "book_002": {
                "title": "1984",
                "author": "George Orwell",
                "genre": "scifi",
                "popularity_score": 0.88,
                "rating": 4.3,
                "page_count": 328,
                "publication_year": 1949,
                "availability": 1.0
            }
        }
        
        for book_id, metadata in sample_books.items():
            bandit.add_book(book_id, metadata)
        
        print(f"✓ Added {len(sample_books)} books successfully")
        return sample_books
        
    except Exception as e:
        print(f"✗ Failed to add books: {e}")
        return {}


def test_recommendations(bandit, books):
    """Test recommendation generation."""
    print("\n--- Testing Recommendations ---")
    
    try:
        user_context = {
            "user_id": "test_user",
            "age": 25,
            "preferred_genre": "fiction",
            "preference_fiction": 0.8,
            "preference_nonfiction": 0.2,
            "preference_mystery": 0.6,
            "preference_romance": 0.4,
            "preference_scifi": 0.7,
            "avg_reading_time": 45,
            "borrow_frequency": 3
        }
        
        session_id = "test_session_001"
        
        # Get recommendations
        recommendations = bandit.get_recommendations(
            user_id=user_context["user_id"],
            session_id=session_id,
            user_context=user_context,
            n_recommendations=2
        )
        
        print(f"✓ Generated {len(recommendations)} recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            book_id = rec['book_id']
            book_info = books[book_id]
            print(f"  {i}. {book_info['title']} (Confidence: {rec['confidence_score']:.3f})")
        
        return recommendations, user_context, session_id
        
    except Exception as e:
        print(f"✗ Failed to generate recommendations: {e}")
        return [], {}, ""


def test_interactions(bandit, recommendations, user_context, session_id):
    """Test interaction recording."""
    print("\n--- Testing Interactions ---")
    
    try:
        # Record some interactions
        for rec in recommendations:
            book_id = rec['book_id']
            
            # Simulate user interaction
            bandit.record_interaction(
                user_id=user_context["user_id"],
                session_id=session_id,
                book_id=book_id,
                action="borrow",
                dwell_time=120.0
            )
            
            print(f"✓ Recorded interaction: {book_id} - borrow")
        
        # Get updated recommendations
        updated_recommendations = bandit.get_recommendations(
            user_id=user_context["user_id"],
            session_id=session_id,
            user_context=user_context,
            n_recommendations=2
        )
        
        print(f"✓ Generated updated recommendations after interactions")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to record interactions: {e}")
        return False


def test_model_persistence(bandit):
    """Test model saving and loading."""
    print("\n--- Testing Model Persistence ---")
    
    try:
        # Save model
        model_file = "test_model.json"
        bandit.bandit.save_model(model_file)
        print("✓ Model saved successfully")
        
        # Create new bandit instance
        config = BanditConfig(
            alpha=1.0,
            feature_dim=20,
            regularization=1.0,
            exploration_rate=0.1,
            decay_rate=0.99,
            min_observations=3
        )
        
        new_bandit = BookRecommenderBandit(config)
        
        # Load model
        new_bandit.bandit.load_model(model_file)
        print("✓ Model loaded successfully")
        
        # Clean up
        if os.path.exists(model_file):
            os.remove(model_file)
            print("✓ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test model persistence: {e}")
        return False


def test_settings():
    """Test settings configuration."""
    print("\n--- Testing Settings ---")
    
    try:
        settings = get_settings()
        print("✓ Settings loaded successfully")
        print(f"  - Database URL: {settings.database_url}")
        print(f"  - Redis Host: {settings.redis_host}")
        print(f"  - Bandit Alpha: {settings.bandit_alpha}")
        print(f"  - Exploration Rate: {settings.exploration_rate}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        return False


def main():
    """Run all tests."""
    print("Book Recommender System - Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Settings
    test_results.append(("Settings", test_settings()))
    
    # Test 2: Bandit initialization
    bandit = test_bandit_initialization()
    test_results.append(("Bandit Initialization", bandit is not None))
    
    if bandit is None:
        print("\n✗ Cannot continue tests due to bandit initialization failure")
        return
    
    # Test 3: Book management
    books = test_book_management(bandit)
    test_results.append(("Book Management", len(books) > 0))
    
    if not books:
        print("\n✗ Cannot continue tests due to book management failure")
        return
    
    # Test 4: Recommendations
    recommendations, user_context, session_id = test_recommendations(bandit, books)
    test_results.append(("Recommendations", len(recommendations) > 0))
    
    # Test 5: Interactions
    if recommendations:
        interaction_success = test_interactions(bandit, recommendations, user_context, session_id)
        test_results.append(("Interactions", interaction_success))
    
    # Test 6: Model persistence
    persistence_success = test_model_persistence(bandit)
    test_results.append(("Model Persistence", persistence_success))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
