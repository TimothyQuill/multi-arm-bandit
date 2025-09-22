"""
Contextual Bandit Model for Book Recommendations

Implements LinUCB (Linear Upper Confidence Bound) algorithm for real-time
recommendation learning with exploration-exploitation trade-off.
Now includes global weights for scalable learning across millions of books.
Includes dynamic arm management for candidate refreshment.
Includes REX-like diversity strategies for improved exploration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
import joblib
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random

from config import BanditConfig, DiversityConfig, ArmRefreshmentConfig
from utils import encode_categorical_feature, encode_library_branch

logger = logging.getLogger(__name__)


class ContextualBandit:
    """
    Contextual Bandit using LinUCB algorithm for book recommendations.
    
    This implementation handles:
    - Real-time feature engineering
    - Exploration-exploitation trade-off
    - Reward updates and model learning
    - Cold-start scenarios
    - Global weights for scalable learning across millions of books
    - Dynamic arm management with refreshment strategies
    - REX-like diversity strategies for improved exploration
    """
    
    def __init__(self, config: BanditConfig):
        self.config = config
        if self.config.arm_refreshment is None:
            self.config.arm_refreshment = ArmRefreshmentConfig()
        if self.config.diversity is None:
            self.config.diversity = DiversityConfig()
            
        self.arms = {}  # Dictionary of available books (arms)
        self.A = {}  # A matrices for each arm
        self.b = {}  # b vectors for each arm
        self.theta = {}  # Parameter vectors for each arm
        self.arm_counts = {}  # Number of times each arm was pulled
        self.last_update = {}  # Last update time for each arm
        
        # Global weights for scalable learning
        self.global_theta = np.zeros(self.config.feature_dim)  # Global parameter vector
        self.global_A = self.config.regularization * np.eye(self.config.feature_dim)  # Global A matrix
        self.global_b = np.zeros(self.config.feature_dim)  # Global b vector
        self.global_count = 0  # Total number of global updates
        self.global_last_update = datetime.now()
        
        # Dynamic arm management
        self.active_arms = set()  # Currently active arms
        self.arm_exposure_count = defaultdict(int)  # How many times each arm has been shown
        self.arm_last_shown = {}  # When each arm was last shown
        self.arm_cooldown_until = {}  # When each arm comes out of cooldown
        self.refreshed_arms = set()  # Arms that have been refreshed
        self.arm_confidence_history = defaultdict(deque)  # Recent confidence scores for each arm
        self.candidate_pool = set()  # Pool of potential candidate arms
        
        # Diversity and exploration tracking
        self.arm_feature_cache = {}  # Cache of arm features for diversity calculations
        self.arm_feature_timestamps = {}  # Timestamps when features were last updated
        self.diversity_clusters = {}  # Clustering information for arms
        self.arm_similarity_matrix = {}  # Precomputed similarity matrix
        self.recent_selections = deque(maxlen=100)  # Recent arm selections for temporal diversity
        self.diversity_scores = defaultdict(float)  # Diversity scores for each arm
        self.exploration_history = defaultdict(list)  # History of exploration decisions
        
        # Session tracking
        self.session_features = {}
        self.user_history = {}
        
        logger.info(f"Initialised Contextual Bandit with config: {config}")
    
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
            
            # Initialize arm management
            self.active_arms.add(arm_id)
            self.arm_exposure_count[arm_id] = 0
            self.arm_last_shown[arm_id] = None
            self.arm_cooldown_until[arm_id] = None
            
            # Initialize diversity tracking
            self.arm_feature_cache[arm_id] = initial_features if initial_features is not None else np.zeros(self.config.feature_dim)
            self.arm_feature_timestamps[arm_id] = datetime.now()
            self.diversity_scores[arm_id] = 1.0  # Start with maximum diversity
            
            logger.info(f"Added new arm: {arm_id}")
        else:
            # Update existing arm features if provided
            if initial_features is not None:
                self.update_arm_features(arm_id, initial_features)
    
    def add_candidate_pool(self, candidate_arm_ids: List[str]):
        """Add arms to the candidate pool for potential refreshment."""
        self.candidate_pool.update(candidate_arm_ids)
        logger.info(f"Added {len(candidate_arm_ids)} arms to candidate pool. Total pool size: {len(self.candidate_pool)}")
    
    def update_arm_features(self, arm_id: str, new_features: np.ndarray):
        """
        Update the cached features for an arm.
        
        Args:
            arm_id: Arm identifier
            new_features: New feature vector
        """
        if arm_id in self.arms:
            # Update the arm's features
            self.arms[arm_id]['features'] = new_features
            
            # Update the feature cache for diversity calculations
            self.arm_feature_cache[arm_id] = new_features.copy()
            
            # Update timestamp
            self.arm_feature_timestamps[arm_id] = datetime.now()
            
            # Invalidate clustering for this arm since features changed
            if arm_id in self.diversity_clusters:
                del self.diversity_clusters[arm_id]
            
            # Reset diversity score to encourage re-evaluation
            self.diversity_scores[arm_id] = 1.0
            
            logger.info(f"Updated features for arm {arm_id}")
        else:
            logger.warning(f"Cannot update features for non-existent arm: {arm_id}")
    
    def _are_features_stale(self, arm_id: str) -> bool:
        """
        Check if an arm's features are stale and need refreshing.
        
        Args:
            arm_id: Arm identifier
            
        Returns:
            True if features are stale
        """
        if arm_id not in self.arm_feature_timestamps:
            return True
        
        hours_since_update = (datetime.now() - self.arm_feature_timestamps[arm_id]).total_seconds() / 3600
        return hours_since_update > self.config.diversity.feature_staleness_hours
    
    def _refresh_stale_features(self, arm_ids: List[str], user_context: Dict[str, Any], 
                               session_context: Dict[str, Any]):
        """
        Refresh stale features for a list of arms.
        
        Args:
            arm_ids: List of arm IDs to check and refresh
            user_context: User context
            session_context: Session context
        """
        for arm_id in arm_ids:
            if self._are_features_stale(arm_id):
                # For now, we'll just update the timestamp to avoid infinite loops
                # In a real implementation, you'd want to fetch fresh metadata
                self.arm_feature_timestamps[arm_id] = datetime.now()
                logger.info(f"Marked features as refreshed for arm {arm_id}")
    
    def refresh_arm_features(self, arm_id: str, user_context: Dict[str, Any], 
                           session_context: Dict[str, Any], book_features: Dict[str, Any]):
        """
        Refresh arm features by re-extracting them from current context and book metadata.
        
        Args:
            arm_id: Arm identifier
            user_context: User context
            session_context: Session context
            book_features: Updated book features
        """
        if arm_id in self.arms:
            # Extract fresh features
            new_features = self.extract_features(user_context, session_context, book_features)
            
            # Update the cached features
            self.update_arm_features(arm_id, new_features)
            
            logger.info(f"Refreshed features for arm {arm_id}")
        else:
            logger.warning(f"Cannot refresh features for non-existent arm: {arm_id}")
    
    def _compute_arm_similarities(self, arm_ids: List[str], context_features: np.ndarray) -> Dict[str, float]:
        """
        Compute similarities between arms and current context.
        
        Args:
            arm_ids: List of arm IDs to compute similarities for
            context_features: Current context features
            
        Returns:
            Dictionary mapping arm_id to similarity score
        """
        similarities = {}
        
        for arm_id in arm_ids:
            if arm_id in self.arm_feature_cache:
                arm_features = self.arm_feature_cache[arm_id]
                # Compute cosine similarity
                similarity = cosine_similarity([context_features], [arm_features])[0][0]
                similarities[arm_id] = similarity
        else:
                similarities[arm_id] = 0.0
        
        return similarities
    
    def _compute_diversity_clusters(self, arm_ids: List[str]):
        """
        Compute diversity clusters for arms using K-means clustering.
        
        Args:
            arm_ids: List of arm IDs to cluster
        """
        if len(arm_ids) < 2:
            return
        
        # Extract features for clustering
        features_matrix = []
        valid_arm_ids = []
        
        for arm_id in arm_ids:
            if arm_id in self.arm_feature_cache:
                features_matrix.append(self.arm_feature_cache[arm_id])
                valid_arm_ids.append(arm_id)
        
        if len(features_matrix) < 2:
            return
        
        features_matrix = np.array(features_matrix)
        
        # Determine number of clusters
        n_clusters = min(self.config.diversity.max_diversity_clusters, len(valid_arm_ids) // 2)
        if n_clusters < 2:
            n_clusters = 2
        
        try:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_matrix)
            
            # Store cluster assignments
            for i, arm_id in enumerate(valid_arm_ids):
                self.diversity_clusters[arm_id] = cluster_labels[i]
                
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            # Fallback: assign random clusters
            for i, arm_id in enumerate(valid_arm_ids):
                self.diversity_clusters[arm_id] = i % n_clusters
    
    def _compute_diversity_scores(self, arm_ids: List[str], context_features: np.ndarray) -> Dict[str, float]:
        """
        Compute diversity scores for arms using multiple strategies.
        
        Args:
            arm_ids: List of arm IDs to compute diversity scores for
            context_features: Current context features
            
        Returns:
            Dictionary mapping arm_id to diversity score
        """
        diversity_scores = {}
        
        # Initialize with base diversity score
        for arm_id in arm_ids:
            diversity_scores[arm_id] = self.diversity_scores.get(arm_id, 1.0)
        
        if not self.config.diversity.use_clustering_diversity:
            return diversity_scores
        
        # Strategy 1: Clustering-based diversity
        if len(arm_ids) > 1:
            self._compute_diversity_clusters(arm_ids)
            
            # Count arms per cluster
            cluster_counts = defaultdict(int)
            for arm_id in arm_ids:
                if arm_id in self.diversity_clusters:
                    cluster_counts[self.diversity_clusters[arm_id]] += 1
            
            # Assign diversity bonus based on cluster size
            for arm_id in arm_ids:
                if arm_id in self.diversity_clusters:
                    cluster_id = self.diversity_clusters[arm_id]
                    cluster_size = cluster_counts[cluster_id]
                    # Smaller clusters get higher diversity scores
                    cluster_diversity = 1.0 / max(cluster_size, 1)
                    diversity_scores[arm_id] *= (1.0 + cluster_diversity * 0.5)
        
        # Strategy 2: Similarity-based diversity penalty
        if self.config.diversity.use_similarity_penalty and len(arm_ids) > 1:
            similarities = self._compute_arm_similarities(arm_ids, context_features)
            
            for arm_id in arm_ids:
                # Penalize arms that are too similar to recently selected arms
                similarity_penalty = 0.0
                for recent_arm, _ in list(self.recent_selections)[-10:]:  # Last 10 selections
                    if recent_arm in similarities:
                        similarity = similarities[recent_arm]
                        if similarity > self.config.diversity.similarity_threshold:
                            similarity_penalty += similarity * 0.3
                
                diversity_scores[arm_id] *= (1.0 - similarity_penalty)
        
        # Strategy 3: Temporal diversity
        if self.config.diversity.use_temporal_diversity:
            current_time = datetime.now()
            for arm_id in arm_ids:
                if arm_id in self.arm_last_shown:
                    hours_since_shown = (current_time - self.arm_last_shown[arm_id]).total_seconds() / 3600
                    if hours_since_shown < self.config.diversity.temporal_window_hours:
                        # Recent arms get lower diversity scores
                        temporal_penalty = 1.0 - (hours_since_shown / self.config.diversity.temporal_window_hours)
                        diversity_scores[arm_id] *= (1.0 - temporal_penalty * 0.4)
        
        # Strategy 4: Exposure-based diversity
        for arm_id in arm_ids:
            exposure_count = self.arm_exposure_count.get(arm_id, 0)
            if exposure_count > 0:
                # Over-exposed arms get lower diversity scores
                exposure_penalty = min(exposure_count / 50.0, 1.0)  # Normalize by 50 exposures
                diversity_scores[arm_id] *= (1.0 - exposure_penalty * 0.3)
        
        return diversity_scores
    
    def _rex_selection(self, arm_scores: List[Tuple[str, float]], n_recommendations: int) -> List[Tuple[str, float]]:
        """
        REX (Randomized Exploration) selection strategy.
        
        Args:
            arm_scores: List of (arm_id, score) tuples
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of selected (arm_id, score) tuples
        """
        if len(arm_scores) <= n_recommendations:
            return arm_scores
        
        # Sort arms by score
        sorted_arms = sorted(arm_scores, key=lambda x: x[1], reverse=True)
        
        # REX strategy: mix top performers with diverse selections
        selected_arms = []
        
        # Select top performers (exploitation)
        top_count = max(1, n_recommendations // 2)
        selected_arms.extend(sorted_arms[:top_count])
        
        # Select diverse arms (exploration)
        remaining_arms = sorted_arms[top_count:]
        if remaining_arms:
            # Use weighted random selection favoring diversity
            diversity_weights = []
            for arm_id, score in remaining_arms:
                diversity_weight = self.diversity_scores.get(arm_id, 1.0)
                # Combine score and diversity
                combined_weight = score * 0.7 + diversity_weight * 0.3
                diversity_weights.append(combined_weight)
            
            # Normalize weights
            total_weight = sum(diversity_weights)
            if total_weight > 0:
                diversity_weights = [w / total_weight for w in diversity_weights]
                
                # Select remaining arms using weighted random selection
                remaining_count = n_recommendations - top_count
                selected_indices = np.random.choice(
                    len(remaining_arms), 
                    size=min(remaining_count, len(remaining_arms)), 
                    replace=False, 
                    p=diversity_weights
                )
                
                for idx in selected_indices:
                    selected_arms.append(remaining_arms[idx])
        
        return selected_arms
    
    def _should_refresh_arms(self) -> bool:
        """
        Determine if arm refreshment should be triggered.
        
        Returns:
            True if refreshment should occur
        """
        if not self.active_arms:
            return True
        
        # Check if max confidence is below threshold
        max_confidence = self._get_max_confidence()
        if max_confidence < self.config.arm_refreshment.min_confidence_threshold:
            logger.info(f"Triggering refreshment: max confidence {max_confidence:.3f} < threshold {self.config.arm_refreshment.min_confidence_threshold}")
            return True
        
        # Check if too many arms are over-exposed
        over_exposed_count = sum(1 for arm_id in self.active_arms 
                               if self.arm_exposure_count[arm_id] >= self.config.arm_refreshment.max_exposure_count)
        
        if over_exposed_count > len(self.active_arms) * 0.5:  # More than 50% over-exposed
            logger.info(f"Triggering refreshment: {over_exposed_count} arms over-exposed")
            return True
        
        return False
    
    def _get_max_confidence(self) -> float:
        """Get the maximum confidence score among active arms."""
        if not self.active_arms:
            return 0.0
        
        max_conf = 0.0
        for arm_id in self.active_arms:
            if arm_id in self.arm_confidence_history and self.arm_confidence_history[arm_id]:
                recent_conf = np.mean(list(self.arm_confidence_history[arm_id])[-5:])  # Average of last 5
                max_conf = max(max_conf, recent_conf)
        
        return max_conf
    
    def _refresh_arms(self, user_context: Dict[str, Any], session_context: Dict[str, Any]):
        """
        Refresh the active arm set by adding new candidates and managing old ones.
        
        Args:
            user_context: User context for feature extraction
            session_context: Session context for feature extraction
        """
        logger.info("Starting arm refreshment process")
        
        # Step 1: Identify arms to deactivate
        arms_to_deactivate = self._identify_arms_to_deactivate()
        
        # Step 2: Deactivate selected arms
        for arm_id in arms_to_deactivate:
            self._deactivate_arm(arm_id)
        
        # Step 3: Select new arms from candidate pool
        new_arms = self._select_new_arms_from_pool(user_context, session_context)
        
        # Step 4: Activate new arms
        for arm_id in new_arms:
            self._activate_arm(arm_id, user_context, session_context)
        
        logger.info(f"Arm refreshment complete: deactivated {len(arms_to_deactivate)}, activated {len(new_arms)}")
    
    def _identify_arms_to_deactivate(self) -> List[str]:
        """
        Identify which arms should be deactivated based on various criteria.
        
        Returns:
            List of arm IDs to deactivate
        """
        arms_to_deactivate = []
        current_time = datetime.now()
        
        for arm_id in list(self.active_arms):
            # Deactivate if over-exposed
            if self.arm_exposure_count[arm_id] >= self.config.arm_refreshment.max_exposure_count:
                arms_to_deactivate.append(arm_id)
                continue
            
            # Deactivate if in cooldown period
            if (arm_id in self.arm_cooldown_until and 
                self.arm_cooldown_until[arm_id] and 
                current_time < self.arm_cooldown_until[arm_id]):
                arms_to_deactivate.append(arm_id)
                continue
            
            # Deactivate if confidence has been consistently low
            if (arm_id in self.arm_confidence_history and 
                len(self.arm_confidence_history[arm_id]) >= 10):
                recent_confidence = np.mean(list(self.arm_confidence_history[arm_id])[-10:])
                if recent_confidence < self.config.arm_refreshment.min_confidence_threshold * 0.5:
                    arms_to_deactivate.append(arm_id)
                    continue
        
        # If we have too many active arms, deactivate the worst performers
        if len(self.active_arms) > self.config.arm_refreshment.max_active_arms:
            # Sort by performance metrics and deactivate worst
            arm_scores = []
            for arm_id in self.active_arms:
                if arm_id not in arms_to_deactivate:
                    score = self._calculate_arm_performance_score(arm_id)
                    arm_scores.append((arm_id, score))
            
            arm_scores.sort(key=lambda x: x[1])  # Sort by score (ascending)
            excess_count = len(self.active_arms) - self.config.arm_refreshment.max_active_arms
            for arm_id, _ in arm_scores[:excess_count]:
                arms_to_deactivate.append(arm_id)
        
        return arms_to_deactivate
    
    def _calculate_arm_performance_score(self, arm_id: str) -> float:
        """
        Calculate a performance score for an arm based on multiple factors.
        
        Args:
            arm_id: Arm identifier
            
        Returns:
            Performance score (lower is worse)
        """
        score = 0.0
        
        # Factor 1: Recent confidence (higher is better)
        if arm_id in self.arm_confidence_history and self.arm_confidence_history[arm_id]:
            recent_conf = np.mean(list(self.arm_confidence_history[arm_id])[-5:])
            score += recent_conf * 0.4
        
        # Factor 2: Observation count (moderate is better - not too few, not too many)
        obs_count = self.arm_counts.get(arm_id, 0)
        if obs_count > 0:
            # Optimal range is around 50-200 observations
            if 50 <= obs_count <= 200:
                score += 0.3
            else:
                score += 0.3 * (1.0 - abs(obs_count - 125) / 125.0)
        
        # Factor 3: Exposure count (lower is better)
        exposure = self.arm_exposure_count.get(arm_id, 0)
        exposure_penalty = min(exposure / self.config.arm_refreshment.max_exposure_count, 1.0)
        score += (1.0 - exposure_penalty) * 0.2
        
        # Factor 4: Recency (more recent is better)
        if arm_id in self.arm_last_shown and self.arm_last_shown[arm_id]:
            hours_since_shown = (datetime.now() - self.arm_last_shown[arm_id]).total_seconds() / 3600
            recency_score = max(0, 1.0 - hours_since_shown / 168.0)  # Decay over 1 week
            score += recency_score * 0.1
        
        return score
    
    def _deactivate_arm(self, arm_id: str):
        """
        Deactivate an arm and put it in cooldown.
        
        Args:
            arm_id: Arm identifier to deactivate
        """
        if arm_id in self.active_arms:
            self.active_arms.remove(arm_id)
            self.refreshed_arms.add(arm_id)
            
            # Set cooldown period
            cooldown_end = datetime.now() + timedelta(hours=self.config.arm_refreshment.cooldown_period_hours)
            self.arm_cooldown_until[arm_id] = cooldown_end
            
            logger.info(f"Deactivated arm {arm_id}, cooldown until {cooldown_end}")
    
    def _select_new_arms_from_pool(self, user_context: Dict[str, Any], 
                                  session_context: Dict[str, Any]) -> List[str]:
        """
        Select new arms from the candidate pool based on user context.
        
        Args:
            user_context: User context for selection
            session_context: Session context for selection
            
        Returns:
            List of selected arm IDs
        """
        if not self.candidate_pool:
            logger.warning("No candidates in pool for refreshment")
            return []
        
        # Filter out arms that are currently active or in cooldown
        available_candidates = []
        current_time = datetime.now()
        
        for arm_id in self.candidate_pool:
            # Skip if already active
            if arm_id in self.active_arms:
                continue
            
            # Skip if in cooldown
            if (arm_id in self.arm_cooldown_until and 
                self.arm_cooldown_until[arm_id] and 
                current_time < self.arm_cooldown_until[arm_id]):
                continue
            
            available_candidates.append(arm_id)
        
        if not available_candidates:
            logger.warning("No available candidates for refreshment")
            return []
        
        # Select arms based on diversity and potential relevance
        selected_arms = self._select_diverse_arms(available_candidates, user_context, session_context)
        
        return selected_arms[:self.config.arm_refreshment.refreshment_batch_size]
    
    def _select_diverse_arms(self, candidates: List[str], user_context: Dict[str, Any], 
                           session_context: Dict[str, Any]) -> List[str]:
        """
        Select diverse arms from candidates to maximize exploration.
        
        Args:
            candidates: List of candidate arm IDs
            user_context: User context
            session_context: Session context
            
        Returns:
            List of selected diverse arm IDs
        """
        if len(candidates) <= self.config.arm_refreshment.refreshment_batch_size:
            return candidates
        
        # For now, use random selection with some preference for arms with features
        # In a more sophisticated implementation, you could use clustering or other diversity metrics
        np.random.shuffle(candidates)
        
        # Prioritize arms that have been seen before (have some data)
        seen_arms = [arm_id for arm_id in candidates if arm_id in self.arms]
        unseen_arms = [arm_id for arm_id in candidates if arm_id not in self.arms]
        
        # Mix seen and unseen arms
        selected = []
        seen_ratio = 0.7  # 70% seen, 30% unseen
        
        seen_count = int(self.config.arm_refreshment.refreshment_batch_size * seen_ratio)
        unseen_count = self.config.arm_refreshment.refreshment_batch_size - seen_count
        
        selected.extend(seen_arms[:seen_count])
        selected.extend(unseen_arms[:unseen_count])
        
        return selected
    
    def _activate_arm(self, arm_id: str, user_context: Dict[str, Any], session_context: Dict[str, Any]):
        """
        Activate an arm and add it to the active set.
        
        Args:
            arm_id: Arm identifier to activate
            user_context: User context
            session_context: Session context
        """
        # Add arm if it doesn't exist
        if arm_id not in self.arms:
            self.add_arm(arm_id)
        
        # Activate the arm
        self.active_arms.add(arm_id)
        
        # Reset exposure count if it was previously refreshed
        if arm_id in self.refreshed_arms:
            self.arm_exposure_count[arm_id] = 0
            self.refreshed_arms.remove(arm_id)
        
        # Clear cooldown
        self.arm_cooldown_until[arm_id] = None
        
        logger.info(f"Activated arm {arm_id}")
    
    def _update_exposure_tracking(self, selected_arms: List[str]):
        """
        Update exposure tracking for selected arms.
        
        Args:
            selected_arms: List of arm IDs that were shown
        """
        current_time = datetime.now()
        
        for arm_id in selected_arms:
            self.arm_exposure_count[arm_id] += 1
            self.arm_last_shown[arm_id] = current_time
            
            # Apply exposure decay for old exposures
            hours_since_last = (current_time - self.arm_last_shown[arm_id]).total_seconds() / 3600
            if hours_since_last > self.config.arm_refreshment.exposure_decay_hours:
                decay_factor = 0.5  # Reduce exposure count by half
                self.arm_exposure_count[arm_id] = int(self.arm_exposure_count[arm_id] * decay_factor)
    
    # Feature extraction moved to BookRecommenderBandit class
    
    # Feature encoding methods moved to BookRecommenderBandit class
    
    def select_arm(self, user_context: Dict[str, Any], session_context: Dict[str, Any], 
                   available_books: List[str], n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Select the best arms (books) to recommend using LinUCB algorithm with global weights, 
        dynamic refreshment, and REX-like diversity strategies.
        
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
        
        # Check if arm refreshment should be triggered
        if self._should_refresh_arms():
            self._refresh_arms(user_context, session_context)
        
        # Filter available books to only include active arms
        active_available_books = [book_id for book_id in available_books if book_id in self.active_arms]
        
        if not active_available_books:
            # If no active arms available, fall back to all available books
            active_available_books = available_books
            logger.warning("No active arms available, using all available books")
        
        # Add new arms if they don't exist
        for book_id in active_available_books:
            if book_id not in self.arms:
                self.add_arm(book_id)
        
        arm_scores = []
        
        # Check for stale features if auto-refresh is enabled
        if self.config.diversity.auto_refresh_features:
            self._refresh_stale_features(active_available_books, user_context, session_context)
        
        for arm_id in active_available_books:
            # Get the arm's stored features (these are the combined user/session/book features)
            arm_features = self.arms[arm_id]['features']
            
            # Use the stored features directly (they should already be the combined context features)
            context_features = arm_features
            
            # Calculate UCB score using both individual and global weights
            if self.arm_counts[arm_id] < self.config.min_observations:
                # Exploration phase - use global weights for cold start
                if self.config.use_global_weights and self.global_count > 0:
                    # Use global weights for cold start
                    global_score = context_features.T @ self.global_theta
                    exploration_bonus = self.config.alpha * np.sqrt(
                        context_features.T @ np.linalg.inv(self.global_A) @ context_features
                    )
                    score = global_score + exploration_bonus
                else:
                    # Random selection if no global weights available
                score = np.random.random()
            else:
                # Exploitation with confidence bounds using individual weights
                A_inv = np.linalg.inv(self.A[arm_id])
                theta = A_inv @ self.b[arm_id]
                self.theta[arm_id] = theta
                
                # Calculate UCB score
                exploration_bonus = self.config.alpha * np.sqrt(
                    context_features.T @ A_inv @ context_features
                )
                exploitation_score = context_features.T @ theta
                
                # Combine with global weights if available
                if self.config.use_global_weights and self.global_count > 0:
                    global_score = context_features.T @ self.global_theta
                    # Weighted combination of individual and global scores
                    individual_weight = min(self.arm_counts[arm_id] / 100.0, 1.0)  # More weight to individual as we get more data
                    global_weight = 1.0 - individual_weight
                    exploitation_score = individual_weight * exploitation_score + global_weight * global_score
                
                score = exploitation_score + exploration_bonus
            
            arm_scores.append((arm_id, score))
            
            # Track confidence for refreshment decisions
            self.arm_confidence_history[arm_id].append(score)
            if len(self.arm_confidence_history[arm_id]) > 20:  # Keep only last 20 scores
                self.arm_confidence_history[arm_id].popleft()
        
        # Apply diversity strategies
        if self.config.diversity.use_rex_strategy and len(arm_scores) > n_recommendations:
            # Compute diversity scores
            diversity_scores = self._compute_diversity_scores(active_available_books, context_features)
            
            # Update diversity scores with decay
            for arm_id, div_score in diversity_scores.items():
                self.diversity_scores[arm_id] = div_score
            
            # Apply diversity weighting to scores
            for i, (arm_id, score) in enumerate(arm_scores):
                diversity_bonus = self.diversity_scores.get(arm_id, 1.0) * self.config.diversity.diversity_weight
                arm_scores[i] = (arm_id, score + diversity_bonus)
        
        # Sort by score and return top recommendations
        arm_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply REX selection strategy
        if (self.config.diversity.use_rex_strategy and 
            np.random.random() < self.config.diversity.rex_probability and 
            len(arm_scores) > n_recommendations):
            selected_arms = self._rex_selection(arm_scores, n_recommendations)
        else:
            # Standard selection with epsilon-greedy exploration
        if np.random.random() < self.config.exploration_rate:
            # Randomly shuffle top recommendations
            np.random.shuffle(arm_scores[:n_recommendations])
            selected_arms = arm_scores[:n_recommendations]
        
        # Update tracking
        selected_arm_ids = [arm_id for arm_id, _ in selected_arms]
        self._update_exposure_tracking(selected_arm_ids)
        
        # Track recent selections for temporal diversity
        for arm_id, score in selected_arms:
            self.recent_selections.append((arm_id, score))
            self.exploration_history[arm_id].append({
                'timestamp': datetime.now(),
                'score': score,
                'context': user_context.get('user_id', 'unknown')
            })
        
        return selected_arms
    
    def update(self, arm_id: str, context_features: np.ndarray, reward: float):
        """
        Update the bandit model with observed reward using both individual and global weights.
        
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
        
        # Update individual arm parameters
        self.A[arm_id] += np.outer(context_features, context_features)
        self.b[arm_id] += reward * context_features
        self.arm_counts[arm_id] += 1
        self.last_update[arm_id] = datetime.now()
        
        # Update individual theta
        try:
            A_inv = np.linalg.inv(self.A[arm_id])
            self.theta[arm_id] = A_inv @ self.b[arm_id]
        except np.linalg.LinAlgError:
            logger.warning(f"Singular matrix encountered for arm {arm_id}")
        
        # Update global weights
        if self.config.use_global_weights:
            self._update_global_weights(context_features, reward)
    
    def _update_global_weights(self, context_features: np.ndarray, reward: float):
        """
        Update global weights using online learning approach.
        
        Args:
            context_features: Feature vector used for selection
            reward: Observed reward
        """
        # Apply decay to global weights based on time
        time_diff = (datetime.now() - self.global_last_update).total_seconds() / 3600  # hours
        if time_diff > 0:
            decay_factor = self.config.global_weight_decay ** time_diff
            self.global_A *= decay_factor
            self.global_b *= decay_factor
        
        # Update global parameters
        self.global_A += np.outer(context_features, context_features)
        self.global_b += reward * context_features
        self.global_count += 1
        self.global_last_update = datetime.now()
        
        # Update global theta
        try:
            global_A_inv = np.linalg.inv(self.global_A)
            self.global_theta = global_A_inv @ self.global_b
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix encountered for global weights")
    
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
        """Get statistics about all arms and global weights."""
        stats = {
            'global': {
                'count': self.global_count,
                'last_update': self.global_last_update,
                'theta_norm': np.linalg.norm(self.global_theta) if self.global_count > 0 else 0.0
            },
            'arm_management': {
                'active_arms_count': len(self.active_arms),
                'candidate_pool_size': len(self.candidate_pool),
                'refreshed_arms_count': len(self.refreshed_arms),
                'max_confidence': self._get_max_confidence()
            },
            'diversity': {
                'diversity_clusters_count': len(set(self.diversity_clusters.values())),
                'recent_selections_count': len(self.recent_selections),
                'avg_diversity_score': np.mean(list(self.diversity_scores.values())) if self.diversity_scores else 0.0
            },
            'arms': {}
        }
        
        for arm_id in self.arms:
            stats['arms'][arm_id] = {
                'count': self.arm_counts[arm_id],
                'last_update': self.last_update[arm_id],
                'theta_norm': np.linalg.norm(self.theta[arm_id]) if arm_id in self.theta else 0.0,
                'exposure_count': self.arm_exposure_count[arm_id],
                'is_active': arm_id in self.active_arms,
                'in_cooldown': (arm_id in self.arm_cooldown_until and 
                              self.arm_cooldown_until[arm_id] and 
                              datetime.now() < self.arm_cooldown_until[arm_id]),
                'performance_score': self._calculate_arm_performance_score(arm_id),
                'diversity_score': self.diversity_scores.get(arm_id, 0.0),
                'cluster_id': self.diversity_clusters.get(arm_id, -1)
            }
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the bandit model to disk including global weights and arm management state."""
        model_data = {
            'config': self.config.__dict__,
            'arms': self.arms,
            'A': {k: v.tolist() for k, v in self.A.items()},
            'b': {k: v.tolist() for k, v in self.b.items()},
            'theta': {k: v.tolist() for k, v in self.theta.items()},
            'arm_counts': self.arm_counts,
            'last_update': {k: v.isoformat() for k, v in self.last_update.items()},
            'user_history': self.user_history,
            # Global weights
            'global_theta': self.global_theta.tolist(),
            'global_A': self.global_A.tolist(),
            'global_b': self.global_b.tolist(),
            'global_count': self.global_count,
            'global_last_update': self.global_last_update.isoformat(),
            # Arm management state
            'active_arms': list(self.active_arms),
            'arm_exposure_count': dict(self.arm_exposure_count),
            'arm_last_shown': {k: v.isoformat() if v else None for k, v in self.arm_last_shown.items()},
            'arm_cooldown_until': {k: v.isoformat() if v else None for k, v in self.arm_cooldown_until.items()},
            'refreshed_arms': list(self.refreshed_arms),
            'candidate_pool': list(self.candidate_pool),
            'arm_confidence_history': {k: list(v) for k, v in self.arm_confidence_history.items()},
            # Diversity state
            'arm_feature_cache': {k: v.tolist() for k, v in self.arm_feature_cache.items()},
            'arm_feature_timestamps': {k: v.isoformat() for k, v in self.arm_feature_timestamps.items()},
            'diversity_clusters': dict(self.diversity_clusters),
            'diversity_scores': dict(self.diversity_scores),
            'recent_selections': [(arm_id, score) for arm_id, score in self.recent_selections],
            'exploration_history': {k: [(h['timestamp'].isoformat(), h['score'], h['context']) 
                                      for h in v] for k, v in self.exploration_history.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the bandit model from disk including global weights and arm management state."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.config = BanditConfig(**model_data['config'])
        if self.config.arm_refreshment is None:
            self.config.arm_refreshment = ArmRefreshmentConfig()
        if self.config.diversity is None:
            self.config.diversity = DiversityConfig()
            
        self.arms = model_data['arms']
        self.A = {k: np.array(v) for k, v in model_data['A'].items()}
        self.b = {k: np.array(v) for k, v in model_data['b'].items()}
        self.theta = {k: np.array(v) for k, v in model_data['theta'].items()}
        self.arm_counts = model_data['arm_counts']
        self.last_update = {k: datetime.fromisoformat(v) for k, v in model_data['last_update'].items()}
        self.user_history = model_data['user_history']
        
        # Load global weights
        self.global_theta = np.array(model_data.get('global_theta', [0.0] * self.config.feature_dim))
        self.global_A = np.array(model_data.get('global_A', np.eye(self.config.feature_dim).tolist()))
        self.global_b = np.array(model_data.get('global_b', [0.0] * self.config.feature_dim))
        self.global_count = model_data.get('global_count', 0)
        self.global_last_update = datetime.fromisoformat(model_data.get('global_last_update', datetime.now().isoformat()))
        
        # Load arm management state
        self.active_arms = set(model_data.get('active_arms', []))
        self.arm_exposure_count = defaultdict(int, model_data.get('arm_exposure_count', {}))
        self.arm_last_shown = {k: datetime.fromisoformat(v) if v else None 
                              for k, v in model_data.get('arm_last_shown', {}).items()}
        self.arm_cooldown_until = {k: datetime.fromisoformat(v) if v else None 
                                  for k, v in model_data.get('arm_cooldown_until', {}).items()}
        self.refreshed_arms = set(model_data.get('refreshed_arms', []))
        self.candidate_pool = set(model_data.get('candidate_pool', []))
        self.arm_confidence_history = {k: deque(v) for k, v in model_data.get('arm_confidence_history', {}).items()}
        
        # Load diversity state
        self.arm_feature_cache = {k: np.array(v) for k, v in model_data.get('arm_feature_cache', {}).items()}
        self.arm_feature_timestamps = {k: datetime.fromisoformat(v) 
                                      for k, v in model_data.get('arm_feature_timestamps', {}).items()}
        self.diversity_clusters = model_data.get('diversity_clusters', {})
        self.diversity_scores = defaultdict(float, model_data.get('diversity_scores', {}))
        self.recent_selections = deque(model_data.get('recent_selections', []), maxlen=100)
        self.exploration_history = defaultdict(list)
        for arm_id, history_list in model_data.get('exploration_history', {}).items():
            for timestamp_str, score, context in history_list:
                self.exploration_history[arm_id].append({
                    'timestamp': datetime.fromisoformat(timestamp_str),
                    'score': score,
                    'context': context
                })
        
        logger.info(f"Model loaded from {filepath}")


class BookRecommenderBandit:
    """
    High-level interface for book recommendations using contextual bandits.
    Combines the bandit model with book metadata and user session management.
    Now supports global weights, dynamic arm management, and REX-like diversity strategies 
    for scalable learning across millions of books.
    """
    
    def __init__(self, config: BanditConfig):
        self.bandit = ContextualBandit(config)
        self.book_metadata = {}  # Book metadata cache
        self.session_manager = SessionManager()
        
    def add_book(self, book_id: str, metadata: Dict[str, Any], user_context: Dict[str, Any] = None, 
                 session_context: Dict[str, Any] = None):
        """Add a book with its metadata."""
        self.book_metadata[book_id] = metadata
        
        # If user and session context are provided, extract full features
        if user_context is not None and session_context is not None:
            features = self.extract_features(user_context, session_context, metadata)
        else:
            # Fallback: create basic features from metadata only
        features = np.zeros(self.bandit.config.feature_dim)
        features[0] = metadata.get('popularity_score', 0.5)
        features[1] = metadata.get('availability', 1.0)
        features[2] = metadata.get('rating', 3.5) / 5.0
        features[3] = metadata.get('page_count', 300) / 1000.0
        features[4] = metadata.get('publication_year', 2010) / 2024.0
        
        self.bandit.add_arm(book_id, features)
    
    def update_book_metadata(self, book_id: str, updated_metadata: Dict[str, Any]):
        """Update book metadata and refresh features."""
        if book_id in self.book_metadata:
            # Update metadata
            self.book_metadata[book_id].update(updated_metadata)
            
            # Extract new features from updated metadata
            new_features = self._extract_book_features(self.book_metadata[book_id])
            
            # Update the bandit's feature cache
            self.bandit.update_arm_features(book_id, new_features)
            
            logger.info(f"Updated metadata and features for book {book_id}")
        else:
            logger.warning(f"Cannot update metadata for non-existent book: {book_id}")
    
    def refresh_book_features(self, book_id: str, user_context: Dict[str, Any], 
                            session_context: Dict[str, Any]):
        """Refresh book features using current context and metadata."""
        if book_id in self.book_metadata:
            # Use current metadata to refresh features
            book_features = self._extract_book_features(self.book_metadata[book_id])
            self.bandit.refresh_arm_features(book_id, user_context, session_context, book_features)
        else:
            logger.warning(f"Cannot refresh features for non-existent book: {book_id}")
    
    def add_candidate_books(self, book_ids: List[str]):
        """Add books to the candidate pool for potential refreshment."""
        self.bandit.add_candidate_pool(book_ids)
    
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
        
        # Categorical user features using utility functions
        # Gender encoding
        gender = user_context.get('gender', 'unknown')
        gender_features = encode_categorical_feature(gender, 'gender')
        features.extend(gender_features)
        
        # Library branch encoding (replaces postcode)
        library_branch = user_context.get('library_branch', 'unknown')
        branch_features = encode_library_branch(library_branch)
        features.extend(branch_features)
        
        # Education level
        education_level = user_context.get('education_level', 'unknown')
        education_features = encode_categorical_feature(education_level, 'education_level')
        features.extend(education_features)
        
        # Occupation category
        occupation = user_context.get('occupation_category', 'unknown')
        occupation_features = encode_categorical_feature(occupation, 'occupation_category')
        features.extend(occupation_features)
        
        # Session features using utility functions
        time_of_day = session_context.get('time_of_day', 'afternoon')
        time_features = encode_categorical_feature(time_of_day, 'time_of_day')
        features.extend(time_features)
        
        device = session_context.get('device', 'desktop')
        device_features = encode_categorical_feature(device, 'device')
        features.extend(device_features)
        
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
        
        # Genre features using utility functions
        book_genre = book_features.get('genre', 'fiction')
        genre_features = encode_categorical_feature(book_genre, 'genre')
        features.extend(genre_features)
        
        # Interaction history features
        user_id = user_context.get('user_id', 'unknown')
        if user_id in self.bandit.user_history:
            recent_interactions = self.bandit.user_history[user_id][-10:]  # Last 10 interactions
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
        if len(feature_vector) < self.bandit.config.feature_dim:
            feature_vector = np.pad(feature_vector, (0, self.bandit.config.feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.bandit.config.feature_dim]
        
        return feature_vector
    
    # Encoding functions moved to utils.py - use those instead
    
    def get_recommendations(self, user_id: str, session_id: str, 
                          user_context: Dict[str, Any], n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get personalized book recommendations."""
        # Get or create session
        session = self.session_manager.get_session(session_id, user_id)
        
        # Get available books
        available_books = list(self.book_metadata.keys())
        
        # Prepare features for each book and update the bandit
        for book_id in available_books:
            if book_id in self.book_metadata:
                # Extract fresh features for this book
                book_features = self.book_metadata[book_id]
                combined_features = self.extract_features(user_context, session, book_features)
                
                # Update the bandit's feature cache
                self.bandit.update_arm_features(book_id, combined_features)
        
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
    
    def get_diversity_statistics(self) -> Dict[str, Any]:
        """Get diversity and exploration statistics."""
        stats = self.bandit.get_arm_statistics()
        return {
            'diversity_metrics': stats['diversity'],
            'arm_management': stats['arm_management'],
            'total_arms': len(stats['arms']),
            'active_arms': stats['arm_management']['active_arms_count'],
            'candidate_pool_size': stats['arm_management']['candidate_pool_size']
        }


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
