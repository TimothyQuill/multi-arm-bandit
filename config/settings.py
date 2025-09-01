"""
Configuration settings for the Book Recommender System

Manages all configuration parameters including:
- Database connections
- Redis cache settings
- Model parameters
- API settings
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database settings
    database_url: str = Field(
        default="postgresql://localhost/bookdb",
        env="DATABASE_URL",
        description="PostgreSQL database connection URL"
    )
    
    # Redis settings
    redis_host: str = Field(
        default="localhost",
        env="REDIS_HOST",
        description="Redis server host"
    )
    redis_port: int = Field(
        default=6379,
        env="REDIS_PORT",
        description="Redis server port"
    )
    redis_password: Optional[str] = Field(
        default=None,
        env="REDIS_PASSWORD",
        description="Redis server password"
    )
    
    # Contextual Bandit settings
    bandit_alpha: float = Field(
        default=1.0,
        env="BANDIT_ALPHA",
        description="Exploration parameter for LinUCB algorithm"
    )
    feature_dim: int = Field(
        default=50,
        env="FEATURE_DIM",
        description="Feature dimension for bandit model"
    )
    regularization: float = Field(
        default=1.0,
        env="REGULARIZATION",
        description="L2 regularization parameter"
    )
    exploration_rate: float = Field(
        default=0.1,
        env="EXPLORATION_RATE",
        description="Epsilon for exploration (0-1)"
    )
    decay_rate: float = Field(
        default=0.99,
        env="DECAY_RATE",
        description="Reward decay rate for temporal effects"
    )
    min_observations: int = Field(
        default=10,
        env="MIN_OBSERVATIONS",
        description="Minimum observations before exploitation"
    )
    
    # Model paths
    content_model_path: Optional[str] = Field(
        default=None,
        env="CONTENT_MODEL_PATH",
        description="Path to content-based model file"
    )
    collaborative_model_path: Optional[str] = Field(
        default=None,
        env="COLLABORATIVE_MODEL_PATH",
        description="Path to collaborative filtering model file"
    )
    
    # Recommendation settings
    min_interactions_for_bandit: int = Field(
        default=5,
        env="MIN_INTERACTIONS_FOR_BANDIT",
        description="Minimum user interactions before using bandit"
    )
    min_interactions_for_hybrid: int = Field(
        default=2,
        env="MIN_INTERACTIONS_FOR_HYBRID",
        description="Minimum user interactions before using hybrid approach"
    )
    
    # Cache settings
    cache_ttl: int = Field(
        default=300,
        env="CACHE_TTL",
        description="Cache time-to-live in seconds"
    )
    
    # API settings
    api_host: str = Field(
        default="0.0.0.0",
        env="API_HOST",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        env="API_PORT",
        description="API server port"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default=None,
        env="LOG_FILE",
        description="Log file path"
    )
    
    # Security settings
    secret_key: str = Field(
        default="your-secret-key-here",
        env="SECRET_KEY",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(
        default="HS256",
        env="ALGORITHM",
        description="JWT algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES",
        description="JWT token expiration time in minutes"
    )
    
    # Performance settings
    max_recommendations: int = Field(
        default=20,
        env="MAX_RECOMMENDATIONS",
        description="Maximum number of recommendations per request"
    )
    batch_size: int = Field(
        default=100,
        env="BATCH_SIZE",
        description="Batch size for model updates"
    )
    
    # Monitoring settings
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS",
        description="Enable performance metrics collection"
    )
    metrics_interval: int = Field(
        default=60,
        env="METRICS_INTERVAL",
        description="Metrics collection interval in seconds"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global _settings
    if _settings is None:
        _settings = Settings()
    
    for key, value in kwargs.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)
    
    return _settings


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    database_url: str = "postgresql://localhost/bookdb_dev"
    redis_host: str = "localhost"
    redis_port: int = 6379


class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    log_level: str = "WARNING"
    cache_ttl: int = 600  # 10 minutes
    min_observations: int = 20
    exploration_rate: float = 0.05


class TestingSettings(Settings):
    """Testing environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    database_url: str = "postgresql://localhost/bookdb_test"
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl: int = 60  # 1 minute


def get_environment_settings(environment: str = None) -> Settings:
    """Get settings for a specific environment."""
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Configuration validation
def validate_settings(settings: Settings) -> bool:
    """Validate configuration settings."""
    errors = []
    
    # Validate database URL
    if not settings.database_url.startswith(("postgresql://", "postgres://")):
        errors.append("Invalid database URL format")
    
    # Validate Redis settings
    if not (0 <= settings.redis_port <= 65535):
        errors.append("Invalid Redis port number")
    
    # Validate bandit parameters
    if not (0 < settings.bandit_alpha <= 10):
        errors.append("Bandit alpha must be between 0 and 10")
    
    if not (0 <= settings.exploration_rate <= 1):
        errors.append("Exploration rate must be between 0 and 1")
    
    if not (0 < settings.decay_rate <= 1):
        errors.append("Decay rate must be between 0 and 1")
    
    if settings.feature_dim <= 0:
        errors.append("Feature dimension must be positive")
    
    # Validate API settings
    if not (1 <= settings.api_port <= 65535):
        errors.append("Invalid API port number")
    
    if settings.max_recommendations <= 0:
        errors.append("Max recommendations must be positive")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    return True


# Default configuration for quick setup
DEFAULT_CONFIG = {
    "database_url": "postgresql://localhost/bookdb",
    "redis_host": "localhost",
    "redis_port": 6379,
    "bandit_alpha": 1.0,
    "feature_dim": 50,
    "regularization": 1.0,
    "exploration_rate": 0.1,
    "decay_rate": 0.99,
    "min_observations": 10,
    "cache_ttl": 300,
    "api_host": "0.0.0.0",
    "api_port": 8000,
    "debug": False,
    "log_level": "INFO",
    "max_recommendations": 20,
    "enable_metrics": True
}


def create_default_config_file(filepath: str = ".env"):
    """Create a default configuration file."""
    config_content = []
    
    for key, value in DEFAULT_CONFIG.items():
        if isinstance(value, str):
            config_content.append(f'{key.upper()}="{value}"')
        else:
            config_content.append(f'{key.upper()}={value}')
    
    config_content.extend([
        "# Add your secret key here",
        'SECRET_KEY="your-secret-key-here"',
        "",
        "# Optional: Model paths",
        "# CONTENT_MODEL_PATH=models/content_model.pkl",
        "# COLLABORATIVE_MODEL_PATH=models/collaborative_model.pkl",
        "",
        "# Optional: Environment",
        "# ENVIRONMENT=development"
    ])
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(config_content))
    
    print(f"Default configuration file created: {filepath}")


if __name__ == "__main__":
    # Create default config file
    create_default_config_file()
    
    # Test settings
    settings = get_settings()
    print("Current settings:")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis Host: {settings.redis_host}:{settings.redis_port}")
    print(f"Bandit Alpha: {settings.bandit_alpha}")
    print(f"Exploration Rate: {settings.exploration_rate}")
    print(f"API Port: {settings.api_port}")
    
    # Validate settings
    try:
        validate_settings(settings)
        print("Settings validation: PASSED")
    except ValueError as e:
        print(f"Settings validation: FAILED - {e}") 