#!/usr/bin/env python3
"""
Startup script for the Book Recommender System

This script starts the FastAPI server with proper configuration
and handles environment setup.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the FastAPI server."""
    print("Book Recommender System - Starting Server")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  No .env file found. Creating from example...")
        example_file = project_root / "env_example.txt"
        if example_file.exists():
            with open(example_file, 'r') as f:
                env_content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("✓ Created .env file from example")
        else:
            print("⚠️  No env_example.txt found. Using default settings.")
    
    # Import after setting up environment
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - API Host: {settings.api_host}")
        print(f"  - API Port: {settings.api_port}")
        print(f"  - Debug Mode: {settings.debug}")
        print(f"  - Log Level: {settings.log_level}")
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        print("Using default settings...")
        
        # Default settings
        host = "0.0.0.0"
        port = 8000
        debug = True
        log_level = "info"
    else:
        host = settings.api_host
        port = settings.api_port
        debug = settings.debug
        log_level = settings.log_level.lower()
    
    print(f"\n🚀 Starting server on {host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the server
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=debug,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
