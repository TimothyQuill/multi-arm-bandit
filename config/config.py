"""
Configuration classes for the book recommendation system.
Supports both PostgreSQL and MySQL databases.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DiversityConfig:
    """Configuration for diversity and exploration strategies."""
    use_rex_strategy: bool = True  # Enable REX-like randomized exploration
    diversity_weight: float = 0.3  # Weight for diversity in final scoring
    rex_probability: float = 0.2  # Probability of using REX strategy
    max_diversity_clusters: int = 10  # Maximum number of diversity clusters
    similarity_threshold: float = 0.8  # Threshold for considering arms similar
    exploration_bonus_factor: float = 1.5  # Multiplier for exploration bonus
    diversity_decay_rate: float = 0.95  # Decay rate for diversity scores
    min_diversity_distance: float = 0.1  # Minimum distance for diversity bonus
    use_clustering_diversity: bool = True  # Use clustering for diversity
    use_similarity_penalty: bool = True  # Penalize similar arms
    use_temporal_diversity: bool = True  # Consider temporal diversity
    temporal_window_hours: int = 24  # Window for temporal diversity
    feature_staleness_hours: int = 24  # Hours after which features are considered stale
    auto_refresh_features: bool = True  # Automatically refresh stale features


@dataclass
class ArmRefreshmentConfig:
    """Configuration for dynamic arm management."""
    min_confidence_threshold: float = 0.3  # Minimum confidence to trigger refreshment
    max_exposure_count: int = 100  # Maximum times an arm can be shown before refreshment
    cooldown_period_hours: int = 24  # Hours before a refreshed arm can be re-selected
    max_active_arms: int = 1000  # Maximum number of active arms to keep in memory
    refreshment_batch_size: int = 50  # Number of new arms to add during refreshment
    confidence_decay_factor: float = 0.95  # Decay factor for confidence over time
    exposure_decay_hours: int = 168  # Hours after which exposure count decays (1 week)


@dataclass
class BanditConfig:
    """Configuration for the contextual bandit model."""
    alpha: float = 1.0  # Exploration parameter
    feature_dim: int = 50  # Feature dimension
    regularization: float = 1.0  # L2 regularization
    exploration_rate: float = 0.1  # Epsilon for exploration
    decay_rate: float = 0.99  # Reward decay rate
    min_observations: int = 10  # Minimum observations before exploitation
    global_weight_decay: float = 0.999  # Decay rate for global weights
    global_learning_rate: float = 0.01  # Learning rate for global weights
    use_global_weights: bool = True  # Whether to use global weights
    arm_refreshment: ArmRefreshmentConfig = None  # Arm refreshment configuration
    diversity: DiversityConfig = None  # Diversity and exploration configuration
    
    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.arm_refreshment is None:
            self.arm_refreshment = ArmRefreshmentConfig()
        if self.diversity is None:
            self.diversity = DiversityConfig()


@dataclass
class FeatureConfig:
    """Configuration for feature extraction and processing."""
    # User feature dimensions
    user_age_normalization: float = 100.0
    user_reading_time_normalization: float = 120.0
    user_borrow_frequency_normalization: float = 10.0
    
    # Book feature dimensions
    book_rating_normalization: float = 5.0
    book_page_count_normalization: float = 1000.0
    book_publication_year_normalization: float = 2024.0
    
    # Session feature dimensions
    session_duration_normalization: float = 60.0
    dwell_time_normalization: float = 300.0
    
    # Library branches will be handled by utils.py and categories.py
    # No need for hardcoded configurations here anymore
    
    def __post_init__(self):
        """Initialise default feature configurations if not provided."""
        # Configuration is now handled by the categories and utils modules
        pass


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    # Database type and connection
    db_type: str = 'mysql'  # 'mysql' or 'postgresql'
    host: str = 'localhost'
    port: int = 3306
    username: str = 'bookdb_user'
    password: str = 'password'
    database: str = 'bookdb'
    charset: str = 'utf8mb4'
    
    # Connection pool settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_pre_ping: bool = True
    pool_recycle: int = 3600
    echo: bool = False
    
    # SSL settings (optional)
    ssl_ca: Optional[str] = None
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_verify_cert: bool = False
    
    def get_database_url(self) -> str:
        """Generate database URL based on configuration."""
        if self.db_type == 'mysql':
            driver = 'pymysql'
            return f'mysql+{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}'
        elif self.db_type == 'postgresql':
            driver = 'psycopg2'
            return f'postgresql+{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}'
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def get_engine_kwargs(self) -> dict:
        """Get SQLAlchemy engine kwargs."""
        kwargs = {
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'pool_pre_ping': self.pool_pre_ping,
            'pool_recycle': self.pool_recycle,
            'echo': self.echo
        }
        
        # Add SSL configuration if provided
        if self.ssl_ca:
            connect_args = {
                'ssl_ca': self.ssl_ca,
                'ssl_verify_cert': self.ssl_verify_cert
            }
            if self.ssl_cert:
                connect_args['ssl_cert'] = self.ssl_cert
            if self.ssl_key:
                connect_args['ssl_key'] = self.ssl_key
            kwargs['connect_args'] = connect_args
            
        return kwargs
