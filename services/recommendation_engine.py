"""
Recommendation Engine Service

Hybrid recommendation system utilising:
1. Cold start: Random popular books
2. Main strategy: Hybrid approach combining collaborative filtering and content-based embeddings
   - Uses FAISS for efficient similarity search
   - Merges embeddings as book features for contextual bandit
   - Applies PCA for dimensionality reduction
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
import faiss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.contextual_bandit import BookRecommenderBandit, BanditConfig

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Hybrid recommendation engine with two main strategies:
    
    1. Cold Start: Random selection from popular books
    2. Hybrid Strategy:
       - Extract user history and compute weighted embeddings from FAISS tables
       - Merge collaborative filtering and content-based embeddings
       - Utilise combined embeddings as book features for contextual bandit
       - Apply PCA for dimensionality reduction
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Embedding dimensions
        self.collaborative_dim = config.get('collaborative_embedding_dim', 50)
        self.content_dim = config.get('content_embedding_dim', 60)
        self.combined_embedding_dim = self.collaborative_dim + self.content_dim
        
        # PCA for feature reduction
        self.feature_pca_components = config.get('pca_components', 50)
        self.feature_pca = PCA(n_components=self.feature_pca_components)
        self.feature_scaler = StandardScaler()
        self.pca_fitted = False
        
        # Initialize contextual bandit with combined embedding dimension
        bandit_config = BanditConfig(
            alpha=config.get('bandit_alpha', 1.0),
            feature_dim=self.feature_pca_components,
            regularization=config.get('regularization', 1.0),
            exploration_rate=config.get('exploration_rate', 0.1),
            decay_rate=config.get('decay_rate', 0.99),
            min_observations=config.get('min_observations', 10)
        )
        
        self.bandit = BookRecommenderBandit(bandit_config)
        
        # FAISS indices for embedding similarity search
        self.collaborative_filter_model = None  # FAISS index
        self.content_filtering_model = None     # FAISS index
        self.book_id_to_index = {}             # Maps book_id to FAISS index
        self.index_to_book_id = {}             # Maps FAISS index to book_id
        
        # Embedding caches
        self.collaborative_embeddings = {}     # book_id -> embedding
        self.content_embeddings = {}           # book_id -> embedding
        
        # Database connections
        self.db_engine = None
        self.redis_client = None
        
        # Performance tracking
        self.metrics = {
            'total_recommendations': 0,
            'hybrid_recommendations': 0,
            'cold_start_recommendations': 0,
            'avg_response_time': 0.0,
            'pca_fit_count': 0
        }
        
        self._initialise_connections()
        self._load_embedding_models()
        
        logger.info("Hybrid Recommendation Engine initialised successfully")
    
    def _initialise_connections(self):
        """Initialise database and cache connections."""
        try:
            # MySQL database connection
            db_url = self.config.get('database_url', 'mysql+pymysql://user:password@localhost/bookdb')
            self.db_engine = create_engine(
                db_url,
                pool_pre_ping=True,  # Enable connection health checks
                pool_recycle=3600,   # Recycle connections every hour
                echo=self.config.get('sql_debug', False)
            )
            
            # Redis connection
            redis_host = self.config.get('redis_host', 'localhost')
            redis_port = self.config.get('redis_port', 6379)
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            
            logger.info("MySQL database and Redis cache connections established")
        except Exception as e:
            logger.error(f"Failed to initialise connections: {e}")
    
    def _load_embedding_models(self):
        """Load FAISS indices for collaborative filtering and content-based embeddings."""
        try:
            # Load collaborative filtering FAISS index
            collaborative_index_path = self.config.get('collaborative_faiss_index_path')
            collaborative_mapping_path = self.config.get('collaborative_mapping_path')
            
            if collaborative_index_path and collaborative_mapping_path:
                self.collaborative_filter_model = faiss.read_index(collaborative_index_path)
                self._load_book_mappings(collaborative_mapping_path)
                logger.info(f"Collaborative filtering FAISS index loaded: {self.collaborative_filter_model.ntotal} embeddings")
            
            # Load content-based FAISS index
            content_index_path = self.config.get('content_faiss_index_path')
            content_mapping_path = self.config.get('content_mapping_path')
            
            if content_index_path and content_mapping_path:
                self.content_filtering_model = faiss.read_index(content_index_path)
                # Assuming same book mapping for now, but could be different
                if not self.book_id_to_index:
                    self._load_book_mappings(content_mapping_path)
                logger.info(f"Content-based FAISS index loaded: {self.content_filtering_model.ntotal} embeddings")
            
            # Load embedding caches for faster access
            self._load_embedding_caches()
                
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
    
    def _load_book_mappings(self, mapping_path: str):
        """Load book ID to FAISS index mappings."""
        try:
            import pickle
            with open(mapping_path, 'rb') as f:
                self.book_id_to_index = pickle.load(f)
            
            # Create reverse mapping
            self.index_to_book_id = {v: k for k, v in self.book_id_to_index.items()}
            logger.info(f"Loaded mappings for {len(self.book_id_to_index)} books")
            
        except Exception as e:
            logger.error(f"Failed to load book mappings from {mapping_path}: {e}")
    
    def _load_embedding_caches(self):
        """Load embeddings into memory caches for faster access."""
        try:
            # Load collaborative embeddings
            if self.collaborative_filter_model:
                for book_id, index in self.book_id_to_index.items():
                    if index < self.collaborative_filter_model.ntotal:
                        embedding = self.collaborative_filter_model.reconstruct(index)
                        self.collaborative_embeddings[book_id] = embedding
                logger.info(f"Cached {len(self.collaborative_embeddings)} collaborative embeddings")
            
            # Load content embeddings
            if self.content_filtering_model:
                for book_id, index in self.book_id_to_index.items():
                    if index < self.content_filtering_model.ntotal:
                        embedding = self.content_filtering_model.reconstruct(index)
                        self.content_embeddings[book_id] = embedding
                logger.info(f"Cached {len(self.content_embeddings)} content embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load embedding caches: {e}")
    
    def get_recommendations(self, user_id: str, session_id: str, 
                          user_context: Dict[str, Any], n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get personalised book recommendations using hybrid approach.
        
        Two strategies:
        1. Cold start: Random popular books for new users
        2. Hybrid: History-based embeddings + contextual bandit for experienced users
        
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
            
            # Determine recommendation strategy (cold start vs hybrid)
            user_history = self._get_user_history(user_id)
            
            if len(user_history) < self.config.get('min_interactions_for_hybrid', 3):
                # Cold start: Random popular books
                recommendations = self._get_cold_start_recommendations(
                    user_id, user_context, n_recommendations
                )
                self.metrics['cold_start_recommendations'] += 1
                strategy = 'cold_start'
            else:
                # Hybrid strategy with embeddings and bandit
                recommendations = self._get_hybrid_recommendations(
                    user_id, session_id, user_context, user_history, n_recommendations
                )
                self.metrics['hybrid_recommendations'] += 1
                strategy = 'hybrid'
            
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
    
    def _get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user interaction history."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT book_id, action, dwell_time, timestamp
                    FROM user_interactions 
                    WHERE user_id = :user_id 
                    ORDER BY timestamp DESC
                    LIMIT 100
                """)
                result = conn.execute(query, {'user_id': user_id})
                return [{'book_id': row[0], 'action': row[1], 'dwell_time': row[2], 'timestamp': row[3]} 
                       for row in result]
        except Exception as e:
            logger.error(f"Failed to get user history: {e}")
            return []
    
    def _get_hybrid_recommendations(self, user_id: str, session_id: str, 
                                  user_context: Dict[str, Any], user_history: List[Dict[str, Any]], 
                                  n_recommendations: int) -> List[Dict[str, Any]]:
        """
        Hybrid recommendation strategy:
        1. Compute weighted user embeddings from history
        2. Find similar books using FAISS
        3. Union collaborative and content-based candidates
        4. Use combined embeddings as book features for contextual bandit
        """
        # Step 1: Compute weighted user embeddings from history
        collaborative_user_embedding = self._compute_user_embedding_from_history(
            user_history, 'collaborative'
        )
        content_user_embedding = self._compute_user_embedding_from_history(
            user_history, 'content'
        )
        
        # Step 2: Find similar books using FAISS (top 1000 each)
        collaborative_candidates = self._find_similar_books(
            collaborative_user_embedding, 'collaborative', k=1000, exclude_books=set(h['book_id'] for h in user_history)
        )
        content_candidates = self._find_similar_books(
            content_user_embedding, 'content', k=1000, exclude_books=set(h['book_id'] for h in user_history)
        )
        
        # Step 3: Union the candidates
        all_candidates = list(set(collaborative_candidates + content_candidates))
        
        if not all_candidates:
            logger.warning(f"No candidates found for user {user_id}, falling back to popular books")
            return self._get_cold_start_recommendations(user_id, user_context, n_recommendations)
        
        # Step 4: Prepare books with combined embeddings for bandit
        self._prepare_books_with_combined_embeddings(all_candidates, user_context, session_id)
        
        # Step 5: Get recommendations from bandit
        session = self.bandit.session_manager.get_session(session_id, user_id)
        recommendations = self.bandit.bandit.select_arm(
            user_context, session, all_candidates, n_recommendations
        )
        
        # Format recommendations
        formatted_recommendations = []
        for book_id, confidence in recommendations:
            recommendation = {
                'book_id': book_id,
                'confidence_score': confidence,
                'source': 'hybrid',
                'metadata': self._get_book_metadata(book_id)
            }
            formatted_recommendations.append(recommendation)
        
        return formatted_recommendations
    
    def _compute_user_embedding_from_history(self, user_history: List[Dict[str, Any]], 
                                           embedding_type: str) -> np.ndarray:
        """
        Compute weighted user embedding from interaction history.
        
        Args:
            user_history: User's interaction history
            embedding_type: 'collaborative' or 'content'
            
        Returns:
            Weighted user embedding vector
        """
        embeddings_cache = (self.collaborative_embeddings if embedding_type == 'collaborative' 
                           else self.content_embeddings)
        embedding_dim = (self.collaborative_dim if embedding_type == 'collaborative' 
                        else self.content_dim)
        
        if not user_history or not embeddings_cache:
            return np.zeros(embedding_dim)
        
        weighted_embeddings = []
        weights = []
        current_time = datetime.now()
        
        for interaction in user_history:
            book_id = interaction['book_id']
            action = interaction['action']
            timestamp = interaction['timestamp']
            
            # Skip if book not in embeddings
            if book_id not in embeddings_cache:
                continue
            
            # Calculate magnitude based on action
            magnitude = {
                'borrow': 1.0,
                'view': 0.5,
                'click': 0.3
            }.get(action, 0.1)
            
            # Calculate linear decay based on time
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            days_ago = (current_time - timestamp).days
            time_decay = max(0.1, 1.0 - (days_ago / 365.0))  # Decay over a year
            
            # Final weight
            weight = magnitude * time_decay
            
            # Add to weighted embeddings
            weighted_embeddings.append(embeddings_cache[book_id])
            weights.append(weight)
        
        if not weighted_embeddings:
            return np.zeros(embedding_dim)
        
        # Compute weighted average
        weighted_embeddings = np.array(weighted_embeddings)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        user_embedding = np.average(weighted_embeddings, axis=0, weights=weights)
        return user_embedding
    
    def _find_similar_books(self, user_embedding: np.ndarray, embedding_type: str, 
                           k: int = 1000, exclude_books: set = None) -> List[str]:
        """
        Find similar books using FAISS similarity search.
        
        Args:
            user_embedding: User's computed embedding
            embedding_type: 'collaborative' or 'content'
            k: Number of similar books to find
            exclude_books: Set of book IDs to exclude
            
        Returns:
            List of similar book IDs
        """
        if exclude_books is None:
            exclude_books = set()
        
        # Choose the appropriate FAISS index
        index = (self.collaborative_filter_model if embedding_type == 'collaborative' 
                else self.content_filtering_model)
        
        if index is None or np.allclose(user_embedding, 0):
            return []
        
        # Search for similar items
        try:
            # Reshape for FAISS search
            query_vector = user_embedding.reshape(1, -1).astype('float32')
            
            # Search for more items than needed to account for exclusions
            search_k = min(k + len(exclude_books) + 100, index.ntotal)
            distances, indices = index.search(query_vector, search_k)
            
            # Convert indices to book IDs and filter exclusions
            similar_books = []
            for idx in indices[0]:
                if idx in self.index_to_book_id:
                    book_id = self.index_to_book_id[idx]
                    if book_id not in exclude_books:
                        similar_books.append(book_id)
                        if len(similar_books) >= k:
                            break
            
            return similar_books
            
        except Exception as e:
            logger.error(f"Error in FAISS search for {embedding_type}: {e}")
            return []
    
    def _prepare_books_with_combined_embeddings(self, book_ids: List[str], 
                                              user_context: Dict[str, Any], session_id: str):
        """
        Prepare books with combined embeddings as features for the bandit.
        
        Args:
            book_ids: List of candidate book IDs
            user_context: User context
            session_id: Session ID
        """
        session = self.bandit.session_manager.get_session(session_id, user_context.get('user_id', 'unknown'))
        
        for book_id in book_ids:
            # Get embeddings for this book
            collaborative_emb = self.collaborative_embeddings.get(book_id, np.zeros(self.collaborative_dim))
            content_emb = self.content_embeddings.get(book_id, np.zeros(self.content_dim))
            
            # Combine embeddings
            combined_embedding = np.concatenate([collaborative_emb, content_emb])
            
            # Get book metadata for traditional features
            book_metadata = self._get_book_metadata(book_id)
            
            # Extract traditional features
            traditional_features = self.bandit.extract_features(user_context, session, book_metadata)
            
            # Combine traditional features with embeddings
            full_features = np.concatenate([traditional_features, combined_embedding])
            
            # Apply PCA if fitted, otherwise collect for fitting
            if self.pca_fitted:
                # Scale and transform
                scaled_features = self.feature_scaler.transform([full_features])
                reduced_features = self.feature_pca.transform(scaled_features)[0]
            else:
                # For the first batch, we'll store features and fit PCA later
                reduced_features = full_features[:self.feature_pca_components]  # Temporary truncation
            
            # Add/update book in bandit with combined features
            self.bandit.add_book(book_id, book_metadata, user_context, session)
            self.bandit.bandit.update_arm_features(book_id, reduced_features)
    
    def _fit_pca_if_needed(self, feature_matrix: np.ndarray):
        """
        Fit PCA on the feature matrix if not already fitted.
        
        Args:
            feature_matrix: Matrix of shape (n_samples, n_features)
        """
        if not self.pca_fitted and len(feature_matrix) > self.feature_pca_components:
            try:
                # Fit scaler and PCA
                scaled_features = self.feature_scaler.fit_transform(feature_matrix)
                self.feature_pca.fit(scaled_features)
                self.pca_fitted = True
                self.metrics['pca_fit_count'] += 1
                logger.info(f"PCA fitted with {len(feature_matrix)} samples, reduced to {self.feature_pca_components} components")
            except Exception as e:
                logger.error(f"Failed to fit PCA: {e}")
    
    def _get_book_metadata(self, book_id: str) -> Dict[str, Any]:
        """Get book metadata from database or cache."""
        try:
            # Try cache first
            if self.redis_client:
                cached_metadata = self.redis_client.get(f"book_metadata:{book_id}")
                if cached_metadata:
                    return json.loads(cached_metadata)
            
            # Query database
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT title, author, genre, publication_year, page_count, 
                           rating, popularity_score, availability
                    FROM books 
                    WHERE book_id = :book_id
                """)
                result = conn.execute(query, {'book_id': book_id})
                row = result.fetchone()
                
                if row:
                    metadata = {
                        'title': row[0],
                        'author': row[1],
                        'genre': row[2],
                        'publication_year': row[3],
                        'page_count': row[4],
                        'rating': row[5],
                        'popularity_score': row[6],
                        'availability': row[7]
                    }
                    
                    # Cache metadata
                    if self.redis_client:
                        self.redis_client.setex(f"book_metadata:{book_id}", 3600, json.dumps(metadata))
                    
                    return metadata
                else:
                    return self._get_default_book_metadata()
                    
        except Exception as e:
            logger.error(f"Failed to get book metadata for {book_id}: {e}")
            return self._get_default_book_metadata()
    
    def _get_default_book_metadata(self) -> Dict[str, Any]:
        """Get default book metadata for missing books."""
        return {
            'title': 'Unknown',
            'author': 'Unknown',
            'genre': 'fiction',
            'publication_year': 2020,
            'page_count': 300,
            'rating': 3.5,
            'popularity_score': 0.5,
            'availability': 1.0
        }
    
    # Old base model methods removed - now using FAISS-based hybrid approach
    
    def _get_cold_start_recommendations(self, user_id: str, user_context: Dict[str, Any], 
                                      n_recommendations: int) -> List[Dict[str, Any]]:
        """Get recommendations for new users (cold start) - random popular books."""
        recommendations = []
        
        # Get popular books
        popular_books = self._get_popular_books(n_recommendations * 3)  # Get more to randomize
        
        if not popular_books:
            # Fallback to any books if no popular books found
            popular_books = self._get_fallback_book_ids(n_recommendations * 2)
        
        # Randomly select from popular books
        import random
        selected_books = random.sample(popular_books, min(n_recommendations, len(popular_books)))
        
        for book_id in selected_books:
            recommendations.append({
                'book_id': book_id,
                'confidence_score': random.uniform(0.4, 0.7),  # Random confidence for cold start
                'source': 'cold_start',
                'reason': 'Random popular book for new user',
                'metadata': self._get_book_metadata(book_id)
            })
        
        return recommendations[:n_recommendations]
    
    def _get_fallback_book_ids(self, n_books: int) -> List[str]:
        """Get fallback book IDs when popular books are not available."""
        try:
            with self.db_engine.connect() as conn:
                # MySQL uses RAND() instead of RANDOM()
                query = text("SELECT book_id FROM books ORDER BY RAND() LIMIT :limit")
                result = conn.execute(query, {'limit': n_books})
                return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get fallback books: {e}")
            return []
    
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
        """Store user interaction in MySQL database."""
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
            logger.error(f"Failed to store interaction in MySQL: {e}")
    
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
        """Get popular books based on recent interactions from MySQL."""
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
            logger.error(f"Failed to get popular books from MySQL: {e}")
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
    
    # Age group methods removed - simplified to random popular books for cold start
    
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
