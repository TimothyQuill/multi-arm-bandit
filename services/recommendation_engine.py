"""
Recommendation Engine Service

Orchestrates the contextual bandit model with base recommendation models
and handles cold-start scenarios for new users and books.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from models.contextual_bandit import BookRecommenderBandit, BanditConfig

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Main recommendation engine that combines contextual bandits with base models.
    
    Features:
    - Contextual bandit for real-time learning
    - Base content/collaborative filtering models
    - Cold-start handling for new users/books
    - Hybrid recommendations
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize contextual bandit
        bandit_config = BanditConfig(
            alpha=config.get('bandit_alpha', 1.0),
            feature_dim=config.get('feature_dim', 50),
            regularization=config.get('regularization', 1.0),
            exploration_rate=config.get('exploration_rate', 0.1),
            decay_rate=config.get('decay_rate', 0.99),
            min_observations=config.get('min_observations', 10)
        )
        
        self.bandit = BookRecommenderBandit(bandit_config)
        
        # Initialize base models (content/collaborative filtering)
        self.content_model = None
        self.collaborative_model = None
        
        # Database connections
        self.db_engine = None
        self.redis_client = None
        
        # Performance tracking
        self.metrics = {
            'total_recommendations': 0,
            'bandit_recommendations': 0,
            'base_model_recommendations': 0,
            'cold_start_recommendations': 0,
            'avg_response_time': 0.0
        }
        
        self._initialize_connections()
        self._load_base_models()
        
        logger.info("Recommendation Engine initialized successfully")
    
    def _initialize_connections(self):
        """Initialize database and cache connections."""
        try:
            # Database connection
            db_url = self.config.get('database_url', 'postgresql://localhost/bookdb')
            self.db_engine = create_engine(db_url)
            
            # Redis connection
            redis_host = self.config.get('redis_host', 'localhost')
            redis_port = self.config.get('redis_port', 6379)
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            
            logger.info("Database and cache connections established")
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
    
    def _load_base_models(self):
        """Load pre-trained base models for content and collaborative filtering."""
        try:
            # Load content-based model
            content_model_path = self.config.get('content_model_path')
            if content_model_path:
                self.content_model = self._load_model(content_model_path)
                logger.info("Content-based model loaded")
            
            # Load collaborative filtering model
            collaborative_model_path = self.config.get('collaborative_model_path')
            if collaborative_model_path:
                self.collaborative_model = self._load_model(collaborative_model_path)
                logger.info("Collaborative filtering model loaded")
                
        except Exception as e:
            logger.error(f"Failed to load base models: {e}")
    
    def _load_model(self, model_path: str):
        """Load a pre-trained model from disk."""
        try:
            import joblib
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    def get_recommendations(self, user_id: str, session_id: str, 
                          user_context: Dict[str, Any], n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get personalized book recommendations using hybrid approach.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_context: User context and preferences
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"recommendations:{user_id}:{session_id}"
            cached_recommendations = self._get_from_cache(cache_key)
            if cached_recommendations:
                logger.info(f"Returning cached recommendations for user {user_id}")
                return cached_recommendations
            
            # Determine recommendation strategy
            strategy = self._determine_strategy(user_id, user_context)
            
            if strategy == 'bandit':
                recommendations = self._get_bandit_recommendations(
                    user_id, session_id, user_context, n_recommendations
                )
                self.metrics['bandit_recommendations'] += 1
                
            elif strategy == 'hybrid':
                recommendations = self._get_hybrid_recommendations(
                    user_id, session_id, user_context, n_recommendations
                )
                self.metrics['base_model_recommendations'] += 1
                
            else:  # cold_start
                recommendations = self._get_cold_start_recommendations(
                    user_id, user_context, n_recommendations
                )
                self.metrics['cold_start_recommendations'] += 1
            
            # Cache recommendations
            self._cache_recommendations(cache_key, recommendations)
            
            # Update metrics
            self.metrics['total_recommendations'] += 1
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_avg_response_time(response_time)
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id} using {strategy} strategy")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _determine_strategy(self, user_id: str, user_context: Dict[str, Any]) -> str:
        """Determine which recommendation strategy to use."""
        # Check if user has enough history for bandit
        user_history_count = self._get_user_interaction_count(user_id)
        
        if user_history_count >= self.config.get('min_interactions_for_bandit', 5):
            return 'bandit'
        elif user_history_count >= self.config.get('min_interactions_for_hybrid', 2):
            return 'hybrid'
        else:
            return 'cold_start'
    
    def _get_bandit_recommendations(self, user_id: str, session_id: str, 
                                  user_context: Dict[str, Any], n_recommendations: int) -> List[Dict[str, Any]]:
        """Get recommendations using contextual bandit."""
        return self.bandit.get_recommendations(user_id, session_id, user_context, n_recommendations)
    
    def _get_hybrid_recommendations(self, user_id: str, session_id: str, 
                                  user_context: Dict[str, Any], n_recommendations: int) -> List[Dict[str, Any]]:
        """Get hybrid recommendations combining bandit and base models."""
        # Get bandit recommendations
        bandit_recs = self.bandit.get_recommendations(user_id, session_id, user_context, n_recommendations)
        
        # Get base model recommendations
        base_recs = self._get_base_model_recommendations(user_id, user_context, n_recommendations)
        
        # Combine recommendations using weighted scoring
        combined_recs = self._combine_recommendations(bandit_recs, base_recs, n_recommendations)
        
        return combined_recs
    
    def _get_base_model_recommendations(self, user_id: str, user_context: Dict[str, Any], 
                                      n_recommendations: int) -> List[Dict[str, Any]]:
        """Get recommendations from base content/collaborative filtering models."""
        recommendations = []
        
        # Try collaborative filtering first
        if self.collaborative_model:
            try:
                cf_recs = self._get_collaborative_recommendations(user_id, n_recommendations)
                recommendations.extend(cf_recs)
            except Exception as e:
                logger.warning(f"Collaborative filtering failed: {e}")
        
        # Fill remaining slots with content-based recommendations
        if len(recommendations) < n_recommendations and self.content_model:
            try:
                content_recs = self._get_content_recommendations(user_context, n_recommendations - len(recommendations))
                recommendations.extend(content_recs)
            except Exception as e:
                logger.warning(f"Content-based filtering failed: {e}")
        
        return recommendations[:n_recommendations]
    
    def _get_collaborative_recommendations(self, user_id: str, n_recommendations: int) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations."""
        # This would integrate with your existing collaborative filtering model
        # For now, return popular books
        popular_books = self._get_popular_books(n_recommendations)
        return [{'book_id': book_id, 'confidence_score': 0.7, 'source': 'collaborative'} 
                for book_id in popular_books]
    
    def _get_content_recommendations(self, user_context: Dict[str, Any], n_recommendations: int) -> List[Dict[str, Any]]:
        """Get content-based recommendations."""
        # This would integrate with your existing content-based model
        # For now, return books based on user preferences
        preferred_genre = user_context.get('preferred_genre', 'fiction')
        genre_books = self._get_books_by_genre(preferred_genre, n_recommendations)
        return [{'book_id': book_id, 'confidence_score': 0.6, 'source': 'content'} 
                for book_id in genre_books]
    
    def _combine_recommendations(self, bandit_recs: List[Dict[str, Any]], 
                               base_recs: List[Dict[str, Any]], n_recommendations: int) -> List[Dict[str, Any]]:
        """Combine recommendations from different sources using weighted scoring."""
        all_recs = {}
        
        # Add bandit recommendations with higher weight
        for rec in bandit_recs:
            book_id = rec['book_id']
            score = rec['confidence_score'] * 0.7  # 70% weight for bandit
            all_recs[book_id] = {
                'book_id': book_id,
                'confidence_score': score,
                'source': 'hybrid',
                'metadata': rec.get('metadata', {})
            }
        
        # Add base model recommendations
        for rec in base_recs:
            book_id = rec['book_id']
            if book_id in all_recs:
                # Average the scores
                all_recs[book_id]['confidence_score'] = (
                    all_recs[book_id]['confidence_score'] + rec['confidence_score'] * 0.3
                ) / 2
            else:
                all_recs[book_id] = {
                    'book_id': book_id,
                    'confidence_score': rec['confidence_score'] * 0.3,
                    'source': 'hybrid'
                }
        
        # Sort by confidence score and return top recommendations
        sorted_recs = sorted(all_recs.values(), key=lambda x: x['confidence_score'], reverse=True)
        return sorted_recs[:n_recommendations]
    
    def _get_cold_start_recommendations(self, user_id: str, user_context: Dict[str, Any], 
                                      n_recommendations: int) -> List[Dict[str, Any]]:
        """Get recommendations for new users (cold start)."""
        recommendations = []
        
        # Popular books
        popular_books = self._get_popular_books(n_recommendations // 2)
        for book_id in popular_books:
            recommendations.append({
                'book_id': book_id,
                'confidence_score': 0.5,
                'source': 'popular',
                'reason': 'Popular among all users'
            })
        
        # Books based on user demographics
        if user_context.get('age'):
            age_group = self._get_age_group(user_context['age'])
            age_books = self._get_books_by_age_group(age_group, n_recommendations - len(recommendations))
            for book_id in age_books:
                recommendations.append({
                    'book_id': book_id,
                    'confidence_score': 0.4,
                    'source': 'demographic',
                    'reason': f'Popular among {age_group} age group'
                })
        
        return recommendations[:n_recommendations]
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> List[Dict[str, Any]]:
        """Get fallback recommendations when all else fails."""
        popular_books = self._get_popular_books(n_recommendations)
        return [{'book_id': book_id, 'confidence_score': 0.3, 'source': 'fallback'} 
                for book_id in popular_books]
    
    def record_interaction(self, user_id: str, session_id: str, book_id: str, 
                          action: str, dwell_time: float = 0.0):
        """Record user interaction and update models."""
        try:
            # Update bandit model
            self.bandit.record_interaction(user_id, session_id, book_id, action, dwell_time)
            
            # Store interaction in database
            self._store_interaction(user_id, book_id, action, dwell_time)
            
            # Invalidate cache
            cache_key = f"recommendations:{user_id}:{session_id}"
            self._invalidate_cache(cache_key)
            
            logger.info(f"Recorded interaction: user={user_id}, book={book_id}, action={action}")
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
    
    def _store_interaction(self, user_id: str, book_id: str, action: str, dwell_time: float):
        """Store user interaction in database."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO user_interactions (user_id, book_id, action, dwell_time, timestamp)
                    VALUES (:user_id, :book_id, :action, :dwell_time, :timestamp)
                """)
                conn.execute(query, {
                    'user_id': user_id,
                    'book_id': book_id,
                    'action': action,
                    'dwell_time': dwell_time,
                    'timestamp': datetime.now()
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
    
    def _get_user_interaction_count(self, user_id: str) -> int:
        """Get the number of interactions for a user."""
        try:
            with self.db_engine.connect() as conn:
                query = text("SELECT COUNT(*) FROM user_interactions WHERE user_id = :user_id")
                result = conn.execute(query, {'user_id': user_id})
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"Failed to get user interaction count: {e}")
            return 0
    
    def _get_popular_books(self, n_books: int) -> List[str]:
        """Get popular books based on recent interactions."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT book_id, COUNT(*) as interaction_count
                    FROM user_interactions
                    WHERE timestamp >= :recent_date
                    GROUP BY book_id
                    ORDER BY interaction_count DESC
                    LIMIT :limit
                """)
                recent_date = datetime.now() - timedelta(days=30)
                result = conn.execute(query, {'recent_date': recent_date, 'limit': n_books})
                return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get popular books: {e}")
            return []
    
    def _get_books_by_genre(self, genre: str, n_books: int) -> List[str]:
        """Get books by genre."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT book_id FROM books 
                    WHERE genre = :genre 
                    ORDER BY popularity_score DESC 
                    LIMIT :limit
                """)
                result = conn.execute(query, {'genre': genre, 'limit': n_books})
                return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get books by genre: {e}")
            return []
    
    def _get_books_by_age_group(self, age_group: str, n_books: int) -> List[str]:
        """Get books popular among specific age group."""
        # This would query books popular among users in the same age group
        # For now, return popular books
        return self._get_popular_books(n_books)
    
    def _get_age_group(self, age: int) -> str:
        """Convert age to age group."""
        if age < 18:
            return 'teen'
        elif age < 25:
            return 'young_adult'
        elif age < 40:
            return 'adult'
        else:
            return 'senior'
    
    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get recommendations from cache."""
        try:
            if self.redis_client:
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    def _cache_recommendations(self, key: str, recommendations: List[Dict[str, Any]]):
        """Cache recommendations."""
        try:
            if self.redis_client:
                ttl = self.config.get('cache_ttl', 300)  # 5 minutes
                self.redis_client.setex(key, ttl, json.dumps(recommendations))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def _invalidate_cache(self, key: str):
        """Invalidate cache entry."""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time metric."""
        total_recs = self.metrics['total_recommendations']
        current_avg = self.metrics['avg_response_time']
        self.metrics['avg_response_time'] = (current_avg * (total_recs - 1) + response_time) / total_recs
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()
    
    def save_model(self, filepath: str):
        """Save the bandit model."""
        self.bandit.bandit.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load the bandit model."""
        self.bandit.bandit.load_model(filepath) 