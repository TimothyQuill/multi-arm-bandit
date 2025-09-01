"""
Contextual Bandit Model for Book Recommendations

Implements LinUCB (Linear Upper Confidence Bound) algorithm for real-time
recommendation learning with exploration-exploitation trade-off.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import joblib
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class BanditConfig:
    """Configuration for the contextual bandit model."""
    alpha: float = 1.0  # Exploration parameter
    feature_dim: int = 50  # Feature dimension
    regularization: float = 1.0  # L2 regularization
    exploration_rate: float = 0.1  # Epsilon for exploration
    decay_rate: float = 0.99  # Reward decay rate
    min_observations: int = 10  # Minimum observations before exploitation


class ContextualBandit:
    """
    Contextual Bandit using LinUCB algorithm for book recommendations.
    
    This implementation handles:
    - Real-time feature engineering
    - Exploration-exploitation trade-off
    - Reward updates and model learning
    - Cold-start scenarios
    """
    
    def __init__(self, config: BanditConfig):
        self.config = config
        self.arms = {}  # Dictionary of available books (arms)
        self.A = {}  # A matrices for each arm
        self.b = {}  # b vectors for each arm
        self.theta = {}  # Parameter vectors for each arm
        self.arm_counts = {}  # Number of times each arm was pulled
        self.last_update = {}  # Last update time for each arm
        
        # Session tracking
        self.session_features = {}
        self.user_history = {}
        
        logger.info(f"Initialized Contextual Bandit with config: {config}")
    
    def add_arm(self, arm_id: str, initial_features: np.ndarray = None):
        """Add a new book (arm) to the bandit."""
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                'id': arm_id,
                'features': initial_features if initial_features is not None else np.zeros(self.config.feature_dim),
                'created_at': datetime.now()
            }
            
            # Initialize LinUCB parameters
            self.A[arm_id] = self.config.regularization * np.eye(self.config.feature_dim)
            self.b[arm_id] = np.zeros(self.config.feature_dim)
            self.theta[arm_id] = np.zeros(self.config.feature_dim)
            self.arm_counts[arm_id] = 0
            self.last_update[arm_id] = datetime.now()
            
            logger.info(f"Added new arm: {arm_id}")
    
    def extract_features(self, user_context: Dict[str, Any], session_context: Dict[str, Any], 
                        book_features: Dict[str, Any]) -> np.ndarray:
        """
        Extract and combine features from user, session, and book context.
        
        Args:
            user_context: User demographics, preferences, history
            session_context: Current session information
            book_features: Book-specific features
            
        Returns:
            Combined feature vector
        """
        features = []
        
        # User features (normalized)
        user_features = [
            user_context.get('age', 25) / 100.0,  # Normalized age
            user_context.get('preference_fiction', 0.5),
            user_context.get('preference_nonfiction', 0.5),
            user_context.get('preference_mystery', 0.5),
            user_context.get('preference_romance', 0.5),
            user_context.get('preference_scifi', 0.5),
            user_context.get('avg_reading_time', 30) / 120.0,  # Normalized reading time
            user_context.get('borrow_frequency', 2) / 10.0,  # Normalized frequency
        ]
        features.extend(user_features)
        
        # Categorical user features
        # Gender encoding (one-hot)
        gender = user_context.get('gender', 'unknown')
        gender_features = {
            'male': [1, 0, 0],
            'female': [0, 1, 0],
            'other': [0, 0, 1],
            'unknown': [0, 0, 0]
        }
        features.extend(gender_features.get(gender, [0, 0, 0]))
        
        # Postcode encoding (geographic clustering)
        postcode = user_context.get('postcode', 'unknown')
        postcode_features = self._encode_postcode(postcode)
        features.extend(postcode_features)
        
        # Education level (categorical)
        education_level = user_context.get('education_level', 'unknown')
        education_categories = ['high_school', 'bachelor', 'master', 'phd', 'other']
        education_features = self._encode_categorical_feature(education_level, education_categories)
        features.extend(education_features)
        
        # Occupation category (categorical)
        occupation = user_context.get('occupation_category', 'unknown')
        occupation_categories = ['student', 'professional', 'retired', 'unemployed', 'other']
        occupation_features = self._encode_categorical_feature(occupation, occupation_categories)
        features.extend(occupation_features)
        
        # Session features
        time_of_day = session_context.get('time_of_day', 'afternoon')
        time_features = {
            'morning': [1, 0, 0, 0],
            'afternoon': [0, 1, 0, 0],
            'evening': [0, 0, 1, 0],
            'night': [0, 0, 0, 1]
        }
        features.extend(time_features.get(time_of_day, [0, 0, 0, 0]))
        
        device = session_context.get('device', 'desktop')
        device_features = {
            'mobile': [1, 0, 0],
            'tablet': [0, 1, 0],
            'desktop': [0, 0, 1]
        }
        features.extend(device_features.get(device, [0, 0, 0]))
        
        # Session duration (normalized)
        session_duration = session_context.get('duration_minutes', 15) / 60.0
        features.append(session_duration)
        
        # Book features
        book_features_list = [
            book_features.get('popularity_score', 0.5),
            book_features.get('availability', 1.0),
            book_features.get('rating', 3.5) / 5.0,
            book_features.get('page_count', 300) / 1000.0,  # Normalized
            book_features.get('publication_year', 2010) / 2024.0,  # Normalized
        ]
        features.extend(book_features_list)
        
        # Genre features (one-hot encoding)
        genres = ['fiction', 'nonfiction', 'mystery', 'romance', 'scifi', 'biography', 'history']
        book_genre = book_features.get('genre', 'fiction')
        genre_features = [1.0 if genre == book_genre else 0.0 for genre in genres]
        features.extend(genre_features)
        
        # Interaction history features
        user_id = user_context.get('user_id', 'unknown')
        if user_id in self.user_history:
            recent_interactions = self.user_history[user_id][-10:]  # Last 10 interactions
            interaction_features = [
                len(recent_interactions) / 10.0,
                sum(1 for i in recent_interactions if i['action'] == 'borrow') / 10.0,
                sum(1 for i in recent_interactions if i['action'] == 'view') / 10.0,
                np.mean([i.get('dwell_time', 0) for i in recent_interactions]) / 300.0,  # Normalized
            ]
        else:
            interaction_features = [0.0, 0.0, 0.0, 0.0]
        features.extend(interaction_features)
        
        # Pad or truncate to feature dimension
        feature_vector = np.array(features)
        if len(feature_vector) < self.config.feature_dim:
            feature_vector = np.pad(feature_vector, (0, self.config.feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.config.feature_dim]
        
        return feature_vector
    
    def _encode_postcode(self, postcode: str) -> List[float]:
        """
        Encode postcode into feature vector using geographic clustering.
        
        Args:
            postcode: Postcode string
            
        Returns:
            Feature vector representing geographic location
        """
        if postcode == 'unknown' or not postcode:
            return [0.0, 0.0, 0.0, 0.0, 0.0]  # Default features for unknown postcode
        
        # Extract first part of postcode (area code)
        area_code = postcode.split()[0] if ' ' in postcode else postcode[:2]
        
        # Geographic region encoding (UK postcode example)
        # You can customize this based on your geographic data
        region_features = {
            # London areas
            'E': [1, 0, 0, 0, 0],  # East London
            'W': [1, 0, 0, 0, 0],  # West London
            'N': [1, 0, 0, 0, 0],  # North London
            'S': [1, 0, 0, 0, 0],  # South London
            'SW': [1, 0, 0, 0, 0], # Southwest London
            'SE': [1, 0, 0, 0, 0], # Southeast London
            'NW': [1, 0, 0, 0, 0], # Northwest London
            'NE': [1, 0, 0, 0, 0], # Northeast London
            'EC': [1, 0, 0, 0, 0], # East Central London
            'WC': [1, 0, 0, 0, 0], # West Central London
            
            # Major cities
            'M': [0, 1, 0, 0, 0],  # Manchester
            'B': [0, 1, 0, 0, 0],  # Birmingham
            'L': [0, 1, 0, 0, 0],  # Liverpool
            'S': [0, 1, 0, 0, 0],  # Sheffield
            'LS': [0, 1, 0, 0, 0], # Leeds
            
            # Scotland
            'G': [0, 0, 1, 0, 0],  # Glasgow
            'EH': [0, 0, 1, 0, 0], # Edinburgh
            'AB': [0, 0, 1, 0, 0], # Aberdeen
            'DD': [0, 0, 1, 0, 0], # Dundee
            
            # Wales
            'CF': [0, 0, 0, 1, 0], # Cardiff
            'SA': [0, 0, 0, 1, 0], # Swansea
            'LL': [0, 0, 0, 1, 0], # Llandudno
            
            # Northern Ireland
            'BT': [0, 0, 0, 0, 1], # Belfast
        }
        
        # Try to match the area code
        for code, features in region_features.items():
            if area_code.upper().startswith(code):
                return features
        
        # If no match found, use a default encoding
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _encode_categorical_feature(self, value: str, categories: List[str], 
                                   unknown_value: str = 'unknown') -> List[float]:
        """
        Generic method to encode categorical features using one-hot encoding.
        
        Args:
            value: The categorical value to encode
            categories: List of possible categories
            unknown_value: Value to use for unknown/missing data
            
        Returns:
            One-hot encoded feature vector
        """
        if value == unknown_value or value not in categories:
            return [0.0] * len(categories)
        
        features = [0.0] * len(categories)
        if value in categories:
            features[categories.index(value)] = 1.0
        
        return features
    
    def select_arm(self, user_context: Dict[str, Any], session_context: Dict[str, Any], 
                   available_books: List[str], n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Select the best arms (books) to recommend using LinUCB algorithm.
        
        Args:
            user_context: User information and preferences
            session_context: Current session information
            available_books: List of available book IDs
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (book_id, confidence_score) tuples
        """
        if not available_books:
            return []
        
        # Add new arms if they don't exist
        for book_id in available_books:
            if book_id not in self.arms:
                self.add_arm(book_id)
        
        arm_scores = []
        
        for arm_id in available_books:
            # Extract features for this specific book
            book_features = self.arms[arm_id]['features']
            context_features = self.extract_features(user_context, session_context, book_features)
            
            # Calculate UCB score
            if self.arm_counts[arm_id] < self.config.min_observations:
                # Exploration phase - random selection
                score = np.random.random()
            else:
                # Exploitation with confidence bounds
                A_inv = np.linalg.inv(self.A[arm_id])
                theta = A_inv @ self.b[arm_id]
                self.theta[arm_id] = theta
                
                # Calculate UCB score
                exploration_bonus = self.config.alpha * np.sqrt(
                    context_features.T @ A_inv @ context_features
                )
                exploitation_score = context_features.T @ theta
                score = exploitation_score + exploration_bonus
            
            arm_scores.append((arm_id, score))
        
        # Sort by score and return top recommendations
        arm_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add some exploration (epsilon-greedy)
        if np.random.random() < self.config.exploration_rate:
            # Randomly shuffle top recommendations
            np.random.shuffle(arm_scores[:n_recommendations])
        
        return arm_scores[:n_recommendations]
    
    def update(self, arm_id: str, context_features: np.ndarray, reward: float):
        """
        Update the bandit model with observed reward.
        
        Args:
            arm_id: ID of the selected book
            context_features: Feature vector used for selection
            reward: Observed reward (e.g., 1.0 for borrow, 0.5 for view, 0.0 for ignore)
        """
        if arm_id not in self.arms:
            logger.warning(f"Attempting to update non-existent arm: {arm_id}")
            return
        
        # Apply reward decay based on time since last update
        if self.last_update[arm_id]:
            time_diff = (datetime.now() - self.last_update[arm_id]).total_seconds() / 3600  # hours
            decay_factor = self.config.decay_rate ** time_diff
            reward *= decay_factor
        
        # Update LinUCB parameters
        self.A[arm_id] += np.outer(context_features, context_features)
        self.b[arm_id] += reward * context_features
        self.arm_counts[arm_id] += 1
        self.last_update[arm_id] = datetime.now()
        
        # Update theta
        try:
            A_inv = np.linalg.inv(self.A[arm_id])
            self.theta[arm_id] = A_inv @ self.b[arm_id]
        except np.linalg.LinAlgError:
            logger.warning(f"Singular matrix encountered for arm {arm_id}")
    
    def record_interaction(self, user_id: str, book_id: str, action: str, 
                          dwell_time: float = 0.0, context_features: np.ndarray = None):
        """
        Record user interaction and update the model.
        
        Args:
            user_id: User identifier
            book_id: Book identifier
            action: User action (borrow, view, ignore)
            dwell_time: Time spent on the book page
            context_features: Context features used for selection
        """
        # Calculate reward based on action
        reward_map = {
            'borrow': 1.0,
            'view': 0.5,
            'click': 0.3,
            'ignore': 0.0
        }
        base_reward = reward_map.get(action, 0.0)
        
        # Adjust reward based on dwell time
        if dwell_time > 0:
            dwell_bonus = min(dwell_time / 300.0, 0.2)  # Max 0.2 bonus for 5+ minutes
            base_reward += dwell_bonus
        
        # Store interaction in user history
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        interaction = {
            'book_id': book_id,
            'action': action,
            'dwell_time': dwell_time,
            'reward': base_reward,
            'timestamp': datetime.now()
        }
        self.user_history[user_id].append(interaction)
        
        # Keep only recent history (last 100 interactions)
        if len(self.user_history[user_id]) > 100:
            self.user_history[user_id] = self.user_history[user_id][-100:]
        
        # Update model if context features are provided
        if context_features is not None:
            self.update(book_id, context_features, base_reward)
        
        logger.info(f"Recorded interaction: user={user_id}, book={book_id}, action={action}, reward={base_reward}")
    
    def get_arm_statistics(self) -> Dict[str, Any]:
        """Get statistics about all arms."""
        stats = {}
        for arm_id in self.arms:
            stats[arm_id] = {
                'count': self.arm_counts[arm_id],
                'last_update': self.last_update[arm_id],
                'theta_norm': np.linalg.norm(self.theta[arm_id]) if arm_id in self.theta else 0.0
            }
        return stats
    
    def save_model(self, filepath: str):
        """Save the bandit model to disk."""
        model_data = {
            'config': self.config,
            'arms': self.arms,
            'A': {k: v.tolist() for k, v in self.A.items()},
            'b': {k: v.tolist() for k, v in self.b.items()},
            'theta': {k: v.tolist() for k, v in self.theta.items()},
            'arm_counts': self.arm_counts,
            'last_update': {k: v.isoformat() for k, v in self.last_update.items()},
            'user_history': self.user_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the bandit model from disk."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.config = BanditConfig(**model_data['config'])
        self.arms = model_data['arms']
        self.A = {k: np.array(v) for k, v in model_data['A'].items()}
        self.b = {k: np.array(v) for k, v in model_data['b'].items()}
        self.theta = {k: np.array(v) for k, v in model_data['theta'].items()}
        self.arm_counts = model_data['arm_counts']
        self.last_update = {k: datetime.fromisoformat(v) for k, v in model_data['last_update'].items()}
        self.user_history = model_data['user_history']
        
        logger.info(f"Model loaded from {filepath}")


class BookRecommenderBandit:
    """
    High-level interface for book recommendations using contextual bandits.
    Combines the bandit model with book metadata and user session management.
    """
    
    def __init__(self, config: BanditConfig):
        self.bandit = ContextualBandit(config)
        self.book_metadata = {}  # Book metadata cache
        self.session_manager = SessionManager()
        
    def add_book(self, book_id: str, metadata: Dict[str, Any]):
        """Add a book with its metadata."""
        self.book_metadata[book_id] = metadata
        
        # Extract features from metadata
        features = self._extract_book_features(metadata)
        self.bandit.add_arm(book_id, features)
    
    def _extract_book_features(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features from book metadata."""
        features = np.zeros(self.bandit.config.feature_dim)
        
        # Basic book features
        features[0] = metadata.get('popularity_score', 0.5)
        features[1] = metadata.get('availability', 1.0)
        features[2] = metadata.get('rating', 3.5) / 5.0
        features[3] = metadata.get('page_count', 300) / 1000.0
        features[4] = metadata.get('publication_year', 2010) / 2024.0
        
        return features
    
    def get_recommendations(self, user_id: str, session_id: str, 
                          user_context: Dict[str, Any], n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get personalized book recommendations."""
        # Get or create session
        session = self.session_manager.get_session(session_id, user_id)
        
        # Get available books
        available_books = list(self.book_metadata.keys())
        
        # Get recommendations from bandit
        recommendations = self.bandit.select_arm(
            user_context, session, available_books, n_recommendations
        )
        
        # Format recommendations with metadata
        formatted_recommendations = []
        for book_id, confidence in recommendations:
            if book_id in self.book_metadata:
                recommendation = {
                    'book_id': book_id,
                    'confidence_score': confidence,
                    'metadata': self.book_metadata[book_id]
                }
                formatted_recommendations.append(recommendation)
        
        return formatted_recommendations
    
    def record_interaction(self, user_id: str, session_id: str, book_id: str, 
                          action: str, dwell_time: float = 0.0):
        """Record user interaction and update the model."""
        # Update session
        self.session_manager.update_session(session_id, user_id, book_id, action)
        
        # Get current session context
        session = self.session_manager.get_session(session_id, user_id)
        
        # Record interaction in bandit
        self.bandit.record_interaction(user_id, book_id, action, dwell_time)


class SessionManager:
    """Manages user sessions and session-based features."""
    
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Get or create a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'user_id': user_id,
                'start_time': datetime.now(),
                'last_activity': datetime.now(),
                'duration_minutes': 0,
                'interactions': [],
                'time_of_day': self._get_time_of_day(),
                'device': 'desktop',  # Default, should be set by client
                'location': 'unknown'  # Default, should be set by client
            }
        
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, user_id: str, book_id: str, action: str):
        """Update session with new interaction."""
        session = self.get_session(session_id, user_id)
        
        interaction = {
            'book_id': book_id,
            'action': action,
            'timestamp': datetime.now()
        }
        session['interactions'].append(interaction)
        session['last_activity'] = datetime.now()
        
        # Update duration
        duration = (session['last_activity'] - session['start_time']).total_seconds() / 60
        session['duration_minutes'] = duration
    
    def _get_time_of_day(self) -> str:
        """Get current time of day category."""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night' 