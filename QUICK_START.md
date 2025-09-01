# Quick Start Guide - Book Recommender System

This guide will get you up and running with the contextual bandit book recommender system in minutes.

## 🚀 Immediate Testing (No Database Required)

### 1. Install Core Dependencies
```bash
pip install numpy pandas scikit-learn fastapi uvicorn pydantic
```

### 2. Test the System
```bash
python test_system.py
```

This will test the core functionality without requiring any external databases.

### 3. Run the Demo
```bash
python demo.py
```

This demonstrates the contextual bandit learning with sample data.

## 🌐 Start the Web API

### 1. Start the Server
```bash
python start_server.py
```

The server will start on `http://localhost:8000`

### 2. Test the API
```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
       "user_context": {
         "user_id": "test_user",
         "age": 25,
         "preferred_genre": "fiction",
         "preference_fiction": 0.8,
         "preference_nonfiction": 0.2
       },
       "n_recommendations": 3
     }'

# Record interaction
curl -X POST "http://localhost:8000/interactions" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "test_user",
       "session_id": "session_001",
       "book_id": "book_001",
       "action": "borrow",
       "dwell_time": 120.0
     }'
```

## 📊 Monitor Performance

### View Metrics
```bash
curl "http://localhost:8000/metrics"
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

## 🔧 Configuration

### Basic Configuration
Copy the example environment file:
```bash
cp env_example.txt .env
```

Key settings to adjust:
- `EXPLORATION_RATE`: Controls exploration vs exploitation (0.1 = 10% exploration)
- `BANDIT_ALPHA`: Exploration parameter for LinUCB (1.0 = balanced)
- `FEATURE_DIM`: Feature vector dimension (50 = default)

### Advanced Configuration
- `DATABASE_URL`: PostgreSQL connection (optional)
- `REDIS_HOST`: Redis cache (optional)
- `CONTENT_MODEL_PATH`: Path to your content-based model
- `COLLABORATIVE_MODEL_PATH`: Path to your collaborative filtering model

## 🧪 Integration with Your Models

### 1. Load Your Existing Models
```python
# In services/recommendation_engine.py
def _load_base_models(self):
    # Load your content-based model
    if self.config.content_model_path:
        self.content_model = joblib.load(self.config.content_model_path)
    
    # Load your collaborative filtering model
    if self.config.collaborative_model_path:
        self.collaborative_model = joblib.load(self.config.collaborative_model_path)
```

### 2. Customize Feature Engineering
```python
# In models/contextual_bandit.py
def extract_features(self, user_context, session_context, book_features):
    # Add your custom features here
    features = []
    
    # Your existing feature extraction logic
    features.extend(your_custom_features)
    
    return np.array(features)
```

## 📈 Understanding the System

### How It Works
1. **Contextual Bandit**: Uses LinUCB algorithm for real-time learning
2. **Feature Engineering**: Combines user, session, and book features
3. **Exploration vs Exploitation**: Balances trying new books vs recommending known good ones
4. **Real-time Updates**: Learns from each user interaction immediately

### Key Components
- `ContextualBandit`: Core LinUCB implementation
- `BookRecommenderBandit`: High-level interface
- `RecommendationEngine`: Orchestrates bandit with base models
- `SessionManager`: Tracks user sessions and behavior

### Learning Process
1. User requests recommendations
2. System extracts features from user context
3. Bandit selects books using LinUCB algorithm
4. User interacts with recommended books
5. System updates model with observed rewards
6. Next recommendations improve based on learning

## 🎯 Next Steps

### For Development
1. Customize feature engineering for your domain
2. Integrate your existing recommendation models
3. Add more sophisticated reward functions
4. Implement A/B testing framework

### For Production
1. Set up PostgreSQL database
2. Configure Redis for caching
3. Add authentication and authorization
4. Implement monitoring and alerting
5. Set up CI/CD pipeline

## 🆘 Troubleshooting

### Common Issues

**Import Errors**: Make sure you're in the project directory and dependencies are installed
```bash
cd book-recommender-system
pip install -r requirements.txt
```

**Configuration Errors**: Check your `.env` file exists and is properly formatted
```bash
cp env_example.txt .env
```

**Model Loading Errors**: Ensure your model files exist and are compatible
```bash
# Test model loading
python -c "import joblib; model = joblib.load('your_model.pkl')"
```

### Getting Help
- Check the logs for detailed error messages
- Run `python test_system.py` to verify core functionality
- Review the API documentation at `http://localhost:8000/docs`

## 📚 Additional Resources

- [LinUCB Algorithm Paper](https://arxiv.org/abs/1003.0146)
- [Contextual Bandits Tutorial](https://banditalgs.com/2016/10/12/contextual-bandits/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Setup Guide](https://www.postgresql.org/docs/current/tutorial-start.html) 