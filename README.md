# Book Recommender System with Contextual Bandits

A real-time book recommendation system that uses contextual bandits to dynamically adapt recommendations based on user behavior, session data, and borrowing history.

## Features

- **Contextual Bandits**: Implements LinUCB algorithm for real-time learning
- **Real-time Adaptation**: Continuously learns from user interactions
- **Session Management**: Tracks user sessions and behavior patterns
- **Multi-modal Features**: Combines content-based and collaborative filtering
- **RESTful API**: FastAPI-based web service for recommendations
- **Redis Caching**: High-performance caching for user sessions and models
- **PostgreSQL Database**: Persistent storage for books, users, and interactions

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Contextual    │
│   Application   │◄──►│   Web Service   │◄──►│   Bandit Model  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Redis Cache   │
                       │   (Sessions)    │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  PostgreSQL DB  │
                       │  (Books/Users)  │
                       └─────────────────┘
```

## Core Components

### 1. Contextual Bandit Model (`models/contextual_bandit.py`)
- Implements LinUCB algorithm for exploration-exploitation
- Handles feature engineering from user context
- Manages arm (book) selection and reward updates

### 2. Session Manager (`services/session_manager.py`)
- Tracks user sessions and behavior patterns
- Manages real-time feature extraction
- Handles session persistence and retrieval

### 3. Recommendation Engine (`services/recommendation_engine.py`)
- Orchestrates recommendation generation
- Combines contextual bandit with base models
- Handles cold-start scenarios

### 4. API Service (`api/main.py`)
- RESTful endpoints for recommendations
- User interaction tracking
- Real-time model updates

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL (for production) or SQLite (for development)
- Redis (optional, for caching)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd book-recommender-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env with your configuration
   nano .env
   ```

4. **Initialize the database** (optional for basic testing)
   ```bash
   python setup_database.py
   ```

5. **Test the system**
   ```bash
   python test_system.py
   ```

6. **Start the server**
   ```bash
   python start_server.py
   ```

### Development Setup

For development with minimal dependencies:

```bash
# Install only core dependencies
pip install numpy pandas scikit-learn fastapi uvicorn pydantic

# Run the demo
python demo.py

# Test the system
python test_system.py
```

### Production Setup

1. **Set up PostgreSQL database**
   ```bash
   python setup_database.py
   ```

2. **Configure Redis** (optional but recommended)
   ```bash
   # Install Redis
   sudo apt-get install redis-server  # Ubuntu/Debian
   brew install redis                 # macOS
   ```

3. **Set production environment variables**
   ```bash
   ENVIRONMENT=production
   DATABASE_URL="postgresql://user:password@host:port/bookdb"
   REDIS_HOST="your-redis-host"
   SECRET_KEY="your-secure-secret-key"
   ```

4. **Start with production settings**
   ```bash
   python start_server.py
   ```

## Usage

### Get Recommendations
```bash
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user123",
       "session_id": "session456",
       "context": {
         "time_of_day": "evening",
         "device": "mobile",
         "location": "home"
       }
     }'
```

### Record User Interaction
```bash
curl -X POST "http://localhost:8000/interactions" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user123",
       "book_id": "book456",
       "action": "borrow",
       "reward": 1.0
     }'
```

## Model Details

### Contextual Bandit Algorithm
- **Algorithm**: LinUCB (Linear Upper Confidence Bound)
- **Features**: User demographics, session context, book features
- **Exploration**: ε-greedy with confidence bounds
- **Update Frequency**: Real-time after each interaction

### Feature Engineering
- **User Features**: Age, preferences, reading history
- **Session Features**: Time, device, location, duration
- **Book Features**: Genre, author, popularity, availability
- **Interaction Features**: Previous actions, dwell time

### Integration with Existing Models

The system is designed to work with your existing content/collaborative filtering models:

1. **Content-Based Model Integration**
   ```python
   # In services/recommendation_engine.py
   def _get_content_recommendations(self, user_context, n_recommendations):
       # Load your existing content model
       if self.content_model:
           return self.content_model.predict(user_context, n_recommendations)
   ```

2. **Collaborative Filtering Integration**
   ```python
   # In services/recommendation_engine.py
   def _get_collaborative_recommendations(self, user_id, n_recommendations):
       # Load your existing collaborative model
       if self.collaborative_model:
           return self.collaborative_model.predict(user_id, n_recommendations)
   ```

3. **Hybrid Approach**
   - The system automatically combines contextual bandit with base models
   - Uses weighted scoring to balance exploration vs exploitation
   - Falls back to base models for cold-start scenarios

## Performance Metrics

- **Click-through Rate (CTR)**
- **Conversion Rate**
- **User Engagement**
- **Exploration vs Exploitation Balance**
- **Model Convergence**

## Configuration

Key configuration parameters in `config/settings.py`:
- `EXPLORATION_RATE`: Controls exploration vs exploitation
- `FEATURE_DIMENSION`: Number of features for bandit model
- `SESSION_TIMEOUT`: Session expiration time
- `CACHE_TTL`: Redis cache time-to-live
- `BATCH_SIZE`: Model update batch size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
