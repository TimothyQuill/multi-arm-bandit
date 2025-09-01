"""
FastAPI Web Service for Book Recommender System

Provides RESTful endpoints for:
- Getting personalized book recommendations
- Recording user interactions
- Model management and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime
import uuid

from services.recommendation_engine import RecommendationEngine
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Book Recommender System API",
    description="Real-time book recommendations using contextual bandits",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommendation engine instance
recommendation_engine = None


# Pydantic models for request/response
class UserContext(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    age: Optional[int] = Field(None, description="User age")
    preferred_genre: Optional[str] = Field(None, description="Preferred book genre")
    preference_fiction: Optional[float] = Field(0.5, description="Preference for fiction (0-1)")
    preference_nonfiction: Optional[float] = Field(0.5, description="Preference for nonfiction (0-1)")
    preference_mystery: Optional[float] = Field(0.5, description="Preference for mystery (0-1)")
    preference_romance: Optional[float] = Field(0.5, description="Preference for romance (0-1)")
    preference_scifi: Optional[float] = Field(0.5, description="Preference for science fiction (0-1)")
    avg_reading_time: Optional[int] = Field(30, description="Average reading time in minutes")
    borrow_frequency: Optional[int] = Field(2, description="Average books borrowed per month")


class SessionContext(BaseModel):
    session_id: Optional[str] = Field(None, description="Session identifier")
    time_of_day: Optional[str] = Field(None, description="Time of day (morning/afternoon/evening/night)")
    device: Optional[str] = Field("desktop", description="Device type (mobile/tablet/desktop)")
    location: Optional[str] = Field("unknown", description="User location")
    duration_minutes: Optional[int] = Field(0, description="Session duration in minutes")


class RecommendationRequest(BaseModel):
    user_context: UserContext
    session_context: Optional[SessionContext] = None
    n_recommendations: int = Field(5, ge=1, le=20, description="Number of recommendations")


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    session_id: str
    strategy_used: str
    response_time: float
    timestamp: datetime


class InteractionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    book_id: str = Field(..., description="Book identifier")
    action: str = Field(..., description="User action (borrow/view/click/ignore)")
    dwell_time: Optional[float] = Field(0.0, description="Time spent on book page in seconds")


class InteractionResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime


class ModelMetrics(BaseModel):
    total_recommendations: int
    bandit_recommendations: int
    base_model_recommendations: int
    cold_start_recommendations: int
    avg_response_time: float


# Dependency to get settings
def get_config():
    return get_settings()


# Dependency to get recommendation engine
def get_recommendation_engine():
    global recommendation_engine
    if recommendation_engine is None:
        config = get_config()
        recommendation_engine = RecommendationEngine(config.dict())
    return recommendation_engine


@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine on startup."""
    global recommendation_engine
    try:
        config = get_config()
        recommendation_engine = RecommendationEngine(config.dict())
        logger.info("Recommendation engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Book Recommender System API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now()
    }


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get personalized book recommendations.
    
    This endpoint uses contextual bandits to provide real-time,
    personalized book recommendations based on user context and session data.
    """
    start_time = datetime.now()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_context.session_id if request.session_context else str(uuid.uuid4())
        
        # Prepare user context
        user_context = request.user_context.dict()
        
        # Prepare session context
        session_context = {}
        if request.session_context:
            session_context = request.session_context.dict()
            session_context['session_id'] = session_id
        
        # Get recommendations
        recommendations = engine.get_recommendations(
            user_id=user_context['user_id'],
            session_id=session_id,
            user_context=user_context,
            n_recommendations=request.n_recommendations
        )
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Log recommendation request
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_context['user_id']}")
        
        return RecommendationResponse(
            recommendations=recommendations,
            session_id=session_id,
            strategy_used="contextual_bandit",  # This could be determined from the engine
            response_time=response_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@app.post("/interactions", response_model=InteractionResponse)
async def record_interaction(
    request: InteractionRequest,
    background_tasks: BackgroundTasks,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Record user interaction with a book.
    
    This endpoint records user interactions (borrow, view, click, ignore)
    and updates the contextual bandit model in real-time.
    """
    try:
        # Record interaction in background to avoid blocking
        background_tasks.add_task(
            engine.record_interaction,
            user_id=request.user_id,
            session_id=request.session_id,
            book_id=request.book_id,
            action=request.action,
            dwell_time=request.dwell_time
        )
        
        logger.info(f"Recorded interaction: user={request.user_id}, book={request.book_id}, action={request.action}")
        
        return InteractionResponse(
            success=True,
            message=f"Interaction recorded successfully: {request.action}",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error recording interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record interaction: {str(e)}")


@app.get("/metrics", response_model=ModelMetrics)
async def get_metrics(engine: RecommendationEngine = Depends(get_recommendation_engine)):
    """
    Get performance metrics for the recommendation system.
    
    Returns metrics including:
    - Total recommendations generated
    - Recommendations by strategy (bandit, base model, cold start)
    - Average response time
    """
    try:
        metrics = engine.get_metrics()
        return ModelMetrics(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.post("/model/save")
async def save_model(
    filepath: str = "models/bandit_model.json",
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Save the current bandit model to disk."""
    try:
        engine.save_model(filepath)
        return {
            "success": True,
            "message": f"Model saved to {filepath}",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")


@app.post("/model/load")
async def load_model(
    filepath: str = "models/bandit_model.json",
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Load a bandit model from disk."""
    try:
        engine.load_model(filepath)
        return {
            "success": True,
            "message": f"Model loaded from {filepath}",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/books/popular")
async def get_popular_books(limit: int = 10):
    """Get popular books based on recent interactions."""
    try:
        engine = get_recommendation_engine()
        popular_books = engine._get_popular_books(limit)
        
        return {
            "books": popular_books,
            "count": len(popular_books),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting popular books: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get popular books: {str(e)}")


@app.get("/books/genre/{genre}")
async def get_books_by_genre(genre: str, limit: int = 10):
    """Get books by genre."""
    try:
        engine = get_recommendation_engine()
        books = engine._get_books_by_genre(genre, limit)
        
        return {
            "genre": genre,
            "books": books,
            "count": len(books),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting books by genre: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get books by genre: {str(e)}")


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        engine = get_recommendation_engine()
        metrics = engine.get_metrics()
        
        return {
            "status": "healthy",
            "recommendation_engine": "operational",
            "total_recommendations": metrics['total_recommendations'],
            "avg_response_time": metrics['avg_response_time'],
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 