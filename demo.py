"""
Demo script for the Book Recommender System

This script demonstrates the contextual bandit recommendation system
with sample data and user interactions.
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any

from models.contextual_bandit import BookRecommenderBandit, BanditConfig
from services.recommendation_engine import RecommendationEngine


def create_sample_books() -> Dict[str, Dict[str, Any]]:
    """Create sample book data for demonstration."""
    books = {
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
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee",
            "genre": "fiction",
            "popularity_score": 0.95,
            "rating": 4.5,
            "page_count": 281,
            "publication_year": 1960,
            "availability": 1.0
        },
        "book_003": {
            "title": "1984",
            "author": "George Orwell",
            "genre": "scifi",
            "popularity_score": 0.88,
            "rating": 4.3,
            "page_count": 328,
            "publication_year": 1949,
            "availability": 1.0
        },
        "book_004": {
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "genre": "romance",
            "popularity_score": 0.85,
            "rating": 4.1,
            "page_count": 432,
            "publication_year": 1813,
            "availability": 1.0
        },
        "book_005": {
            "title": "The Hobbit",
            "author": "J.R.R. Tolkien",
            "genre": "fiction",
            "popularity_score": 0.92,
            "rating": 4.4,
            "page_count": 310,
            "publication_year": 1937,
            "availability": 1.0
        },
        "book_006": {
            "title": "A Brief History of Time",
            "author": "Stephen Hawking",
            "genre": "nonfiction",
            "popularity_score": 0.78,
            "rating": 4.0,
            "page_count": 256,
            "publication_year": 1988,
            "availability": 1.0
        },
        "book_007": {
            "title": "The Da Vinci Code",
            "author": "Dan Brown",
            "genre": "mystery",
            "popularity_score": 0.82,
            "rating": 3.8,
            "page_count": 689,
            "publication_year": 2003,
            "availability": 1.0
        },
        "book_008": {
            "title": "The Alchemist",
            "author": "Paulo Coelho",
            "genre": "fiction",
            "popularity_score": 0.87,
            "rating": 3.9,
            "page_count": 208,
            "publication_year": 1988,
            "availability": 1.0
        },
        "book_009": {
            "title": "Sapiens",
            "author": "Yuval Noah Harari",
            "genre": "nonfiction",
            "popularity_score": 0.91,
            "rating": 4.3,
            "page_count": 443,
            "publication_year": 2011,
            "availability": 1.0
        },
        "book_010": {
            "title": "The Martian",
            "author": "Andy Weir",
            "genre": "scifi",
            "popularity_score": 0.89,
            "rating": 4.2,
            "page_count": 369,
            "publication_year": 2011,
            "availability": 1.0
        }
    }
    return books


def create_sample_users() -> Dict[str, Dict[str, Any]]:
    """Create sample user profiles."""
    users = {
        "user_001": {
            "age": 25,
            "preferred_genre": "fiction",
            "preference_fiction": 0.8,
            "preference_nonfiction": 0.2,
            "preference_mystery": 0.6,
            "preference_romance": 0.4,
            "preference_scifi": 0.7,
            "avg_reading_time": 45,
            "borrow_frequency": 3
        },
        "user_002": {
            "age": 35,
            "preferred_genre": "nonfiction",
            "preference_fiction": 0.3,
            "preference_nonfiction": 0.9,
            "preference_mystery": 0.4,
            "preference_romance": 0.2,
            "preference_scifi": 0.5,
            "avg_reading_time": 60,
            "borrow_frequency": 2
        },
        "user_003": {
            "age": 22,
            "preferred_genre": "scifi",
            "preference_fiction": 0.7,
            "preference_nonfiction": 0.4,
            "preference_mystery": 0.5,
            "preference_romance": 0.3,
            "preference_scifi": 0.9,
            "avg_reading_time": 30,
            "borrow_frequency": 4
        }
    }
    return users


def simulate_user_session(user_id: str, user_profile: Dict[str, Any], 
                         recommender: BookRecommenderBandit, books: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate a user session with recommendations and interactions."""
    print(f"\n=== Simulating session for {user_id} ===")
    print(f"User profile: {user_profile['preferred_genre']} lover, age {user_profile['age']}")
    
    session_id = f"session_{user_id}_{int(time.time())}"
    interactions = []
    
    # Get initial recommendations
    recommendations = recommender.get_recommendations(
        user_id=user_id,
        session_id=session_id,
        user_context=user_profile,
        n_recommendations=5
    )
    
    print(f"\nInitial recommendations:")
    for i, rec in enumerate(recommendations, 1):
        book_id = rec['book_id']
        book_info = books[book_id]
        print(f"{i}. {book_info['title']} by {book_info['author']} (Confidence: {rec['confidence_score']:.3f})")
    
    # Simulate user interactions
    for rec in recommendations[:3]:  # User interacts with top 3 recommendations
        book_id = rec['book_id']
        book_info = books[book_id]
        
        # Simulate user behavior based on preferences
        if book_info['genre'] == user_profile['preferred_genre']:
            action = random.choices(['borrow', 'view'], weights=[0.7, 0.3])[0]
            dwell_time = random.uniform(30, 180)  # 30 seconds to 3 minutes
        else:
            action = random.choices(['view', 'ignore'], weights=[0.6, 0.4])[0]
            dwell_time = random.uniform(10, 60)  # 10 seconds to 1 minute
        
        # Record interaction
        recommender.record_interaction(
            user_id=user_id,
            session_id=session_id,
            book_id=book_id,
            action=action,
            dwell_time=dwell_time
        )
        
        interactions.append({
            'book_id': book_id,
            'action': action,
            'dwell_time': dwell_time,
            'book_title': book_info['title']
        })
        
        print(f"  → {action.capitalize()}ed '{book_info['title']}' (dwell: {dwell_time:.1f}s)")
    
    return interactions


def demonstrate_learning_progression(recommender: BookRecommenderBandit, books: Dict[str, Dict[str, Any]]):
    """Demonstrate how the model learns over multiple sessions."""
    print("\n" + "="*60)
    print("DEMONSTRATING LEARNING PROGRESSION")
    print("="*60)
    
    user_id = "demo_user"
    user_profile = {
        "age": 28,
        "preferred_genre": "fiction",
        "preference_fiction": 0.8,
        "preference_nonfiction": 0.2,
        "preference_mystery": 0.6,
        "preference_romance": 0.4,
        "preference_scifi": 0.7,
        "avg_reading_time": 45,
        "borrow_frequency": 3
    }
    
    # Track recommendation quality over sessions
    session_results = []
    
    for session_num in range(1, 6):
        print(f"\n--- Session {session_num} ---")
        
        # Get recommendations
        session_id = f"session_{session_num}"
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            session_id=session_id,
            user_context=user_profile,
            n_recommendations=3
        )
        
        # Calculate recommendation quality (based on genre match)
        fiction_books = sum(1 for rec in recommendations if books[rec['book_id']]['genre'] == 'fiction')
        quality_score = fiction_books / len(recommendations)
        
        print(f"Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            book_info = books[rec['book_id']]
            print(f"  {i}. {book_info['title']} ({book_info['genre']}) - Confidence: {rec['confidence_score']:.3f}")
        
        print(f"Quality Score: {quality_score:.2f} (fiction books: {fiction_books}/{len(recommendations)})")
        
        # Simulate interactions
        for rec in recommendations:
            book_id = rec['book_id']
            book_info = books[book_id]
            
            # Higher probability of positive interaction for preferred genre
            if book_info['genre'] == 'fiction':
                action = random.choices(['borrow', 'view'], weights=[0.8, 0.2])[0]
            else:
                action = random.choices(['view', 'ignore'], weights=[0.4, 0.6])[0]
            
            dwell_time = random.uniform(20, 120)
            
            recommender.record_interaction(
                user_id=user_id,
                session_id=session_id,
                book_id=book_id,
                action=action,
                dwell_time=dwell_time
            )
            
            print(f"    → {action.capitalize()}ed '{book_info['title']}'")
        
        session_results.append({
            'session': session_num,
            'quality_score': quality_score,
            'avg_confidence': np.mean([rec['confidence_score'] for rec in recommendations])
        })
    
    # Show learning progression
    print(f"\n--- Learning Progression Summary ---")
    for result in session_results:
        print(f"Session {result['session']}: Quality={result['quality_score']:.2f}, Avg Confidence={result['avg_confidence']:.3f}")


def demonstrate_exploration_vs_exploitation(recommender: BookRecommenderBandit, books: Dict[str, Dict[str, Any]]):
    """Demonstrate exploration vs exploitation behavior."""
    print("\n" + "="*60)
    print("DEMONSTRATING EXPLORATION VS EXPLOITATION")
    print("="*60)
    
    user_id = "exploration_user"
    user_profile = {
        "age": 30,
        "preferred_genre": "mystery",
        "preference_fiction": 0.6,
        "preference_nonfiction": 0.4,
        "preference_mystery": 0.9,
        "preference_romance": 0.2,
        "preference_scifi": 0.5,
        "avg_reading_time": 40,
        "borrow_frequency": 2
    }
    
    # Track exploration vs exploitation
    exploration_count = 0
    exploitation_count = 0
    
    for session_num in range(1, 11):
        session_id = f"exploration_session_{session_num}"
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            session_id=session_id,
            user_context=user_profile,
            n_recommendations=2
        )
        
        # Determine if this was exploration or exploitation
        mystery_books = sum(1 for rec in recommendations if books[rec['book_id']]['genre'] == 'mystery')
        if mystery_books >= len(recommendations) * 0.7:  # 70% or more mystery books
            exploitation_count += 1
            strategy = "EXPLOITATION"
        else:
            exploration_count += 1
            strategy = "EXPLORATION"
        
        print(f"Session {session_num} ({strategy}): ", end="")
        for rec in recommendations:
            book_info = books[rec['book_id']]
            print(f"{book_info['genre']} ", end="")
        print(f"(Confidence: {np.mean([rec['confidence_score'] for rec in recommendations]):.3f})")
        
        # Simulate interaction
        for rec in recommendations:
            book_id = rec['book_id']
            action = random.choice(['borrow', 'view'])
            dwell_time = random.uniform(15, 90)
            
            recommender.record_interaction(
                user_id=user_id,
                session_id=session_id,
                book_id=book_id,
                action=action,
                dwell_time=dwell_time
            )
    
    print(f"\nExploration vs Exploitation Summary:")
    print(f"Exploration sessions: {exploration_count}")
    print(f"Exploitation sessions: {exploitation_count}")
    print(f"Exploration rate: {exploration_count / (exploration_count + exploitation_count):.2f}")


def main():
    """Main demo function."""
    print("Book Recommender System - Contextual Bandit Demo")
    print("=" * 60)
    
    # Initialize the recommendation system
    config = BanditConfig(
        alpha=1.0,
        feature_dim=50,
        regularization=1.0,
        exploration_rate=0.2,  # Higher exploration for demo
        decay_rate=0.99,
        min_observations=3
    )
    
    recommender = BookRecommenderBandit(config)
    
    # Load sample data
    books = create_sample_books()
    users = create_sample_users()
    
    print(f"Loaded {len(books)} books and {len(users)} users")
    
    # Add books to the recommender
    for book_id, book_info in books.items():
        recommender.add_book(book_id, book_info)
    
    print("Books added to recommendation system")
    
    # Demo 1: Basic user sessions
    print("\n" + "="*60)
    print("DEMO 1: BASIC USER SESSIONS")
    print("="*60)
    
    for user_id, user_profile in users.items():
        simulate_user_session(user_id, user_profile, recommender, books)
    
    # Demo 2: Learning progression
    demonstrate_learning_progression(recommender, books)
    
    # Demo 3: Exploration vs exploitation
    demonstrate_exploration_vs_exploitation(recommender, books)
    
    # Demo 4: Model statistics
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    
    stats = recommender.bandit.get_arm_statistics()
    print(f"Total books (arms): {len(stats)}")
    
    # Show top books by interaction count
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)
    print("\nTop books by interaction count:")
    for i, (book_id, stat) in enumerate(sorted_stats[:5], 1):
        book_info = books[book_id]
        print(f"{i}. {book_info['title']} - {stat['count']} interactions")
    
    # Save model
    print("\nSaving model...")
    recommender.bandit.save_model("demo_model.json")
    print("Model saved to demo_model.json")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main() 