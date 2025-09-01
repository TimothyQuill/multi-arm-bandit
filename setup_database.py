"""
Database Setup Script for Book Recommender System

Creates the necessary database tables for:
- Books and metadata
- Users and profiles
- User interactions
- Session data
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database(db_name: str, host: str = "localhost", port: int = 5432, 
                   user: str = "postgres", password: str = ""):
    """Create the database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Database '{db_name}' created successfully")
        else:
            logger.info(f"Database '{db_name}' already exists")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


def create_tables(db_url: str):
    """Create all necessary tables."""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Create books table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS books (
                book_id VARCHAR(50) PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                author VARCHAR(255) NOT NULL,
                genre VARCHAR(50),
                popularity_score DECIMAL(3,2) DEFAULT 0.5,
                rating DECIMAL(3,2) DEFAULT 0.0,
                page_count INTEGER,
                publication_year INTEGER,
                availability BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(50) PRIMARY KEY,
                age INTEGER,
                preferred_genre VARCHAR(50),
                preference_fiction DECIMAL(3,2) DEFAULT 0.5,
                preference_nonfiction DECIMAL(3,2) DEFAULT 0.5,
                preference_mystery DECIMAL(3,2) DEFAULT 0.5,
                preference_romance DECIMAL(3,2) DEFAULT 0.5,
                preference_scifi DECIMAL(3,2) DEFAULT 0.5,
                avg_reading_time INTEGER DEFAULT 30,
                borrow_frequency INTEGER DEFAULT 2,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user_interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                interaction_id SERIAL PRIMARY KEY,
                user_id VARCHAR(50) NOT NULL,
                book_id VARCHAR(50) NOT NULL,
                action VARCHAR(20) NOT NULL,
                dwell_time DECIMAL(10,2) DEFAULT 0.0,
                reward DECIMAL(3,2) DEFAULT 0.0,
                session_id VARCHAR(100),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
            )
        """)
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(100) PRIMARY KEY,
                user_id VARCHAR(50) NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_minutes INTEGER DEFAULT 0,
                time_of_day VARCHAR(20),
                device VARCHAR(20),
                location VARCHAR(100),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id 
            ON user_interactions(user_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_interactions_book_id 
            ON user_interactions(book_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_interactions_timestamp 
            ON user_interactions(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_books_genre 
            ON books(genre)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_books_popularity 
            ON books(popularity_score DESC)
        """)
        
        # Create function to update updated_at timestamp
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        # Create triggers for updated_at
        cursor.execute("""
            CREATE TRIGGER update_books_updated_at 
            BEFORE UPDATE ON books 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
        
        cursor.execute("""
            CREATE TRIGGER update_users_updated_at 
            BEFORE UPDATE ON users 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
        
        conn.commit()
        logger.info("All tables created successfully")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def insert_sample_data(db_url: str):
    """Insert sample data for testing."""
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Sample books
        sample_books = [
            ("book_001", "The Great Gatsby", "F. Scott Fitzgerald", "fiction", 0.9, 4.2, 180, 1925),
            ("book_002", "To Kill a Mockingbird", "Harper Lee", "fiction", 0.95, 4.5, 281, 1960),
            ("book_003", "1984", "George Orwell", "scifi", 0.88, 4.3, 328, 1949),
            ("book_004", "Pride and Prejudice", "Jane Austen", "romance", 0.85, 4.1, 432, 1813),
            ("book_005", "The Hobbit", "J.R.R. Tolkien", "fiction", 0.92, 4.4, 310, 1937),
            ("book_006", "A Brief History of Time", "Stephen Hawking", "nonfiction", 0.78, 4.0, 256, 1988),
            ("book_007", "The Da Vinci Code", "Dan Brown", "mystery", 0.82, 3.8, 689, 2003),
            ("book_008", "The Alchemist", "Paulo Coelho", "fiction", 0.87, 3.9, 208, 1988),
            ("book_009", "Sapiens", "Yuval Noah Harari", "nonfiction", 0.91, 4.3, 443, 2011),
            ("book_010", "The Martian", "Andy Weir", "scifi", 0.89, 4.2, 369, 2011)
        ]
        
        cursor.executemany("""
            INSERT INTO books (book_id, title, author, genre, popularity_score, rating, page_count, publication_year)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (book_id) DO NOTHING
        """, sample_books)
        
        # Sample users
        sample_users = [
            ("user_001", 25, "fiction", 0.8, 0.2, 0.6, 0.4, 0.7, 45, 3),
            ("user_002", 35, "nonfiction", 0.3, 0.9, 0.4, 0.2, 0.5, 60, 2),
            ("user_003", 22, "scifi", 0.7, 0.4, 0.5, 0.3, 0.9, 30, 4)
        ]
        
        cursor.executemany("""
            INSERT INTO users (user_id, age, preferred_genre, preference_fiction, preference_nonfiction, 
                              preference_mystery, preference_romance, preference_scifi, avg_reading_time, borrow_frequency)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING
        """, sample_users)
        
        conn.commit()
        logger.info("Sample data inserted successfully")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error inserting sample data: {e}")
        raise


def main():
    """Main setup function."""
    print("Book Recommender System - Database Setup")
    print("=" * 50)
    
    # Database configuration
    db_name = "bookdb"
    host = "localhost"
    port = 5432
    user = "postgres"
    password = ""  # Set your password here
    
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    
    try:
        # Create database
        print("Creating database...")
        create_database(db_name, host, port, user, password)
        
        # Create tables
        print("Creating tables...")
        create_tables(db_url)
        
        # Insert sample data
        print("Inserting sample data...")
        insert_sample_data(db_url)
        
        print("\nDatabase setup completed successfully!")
        print(f"Database URL: {db_url}")
        
    except Exception as e:
        print(f"Database setup failed: {e}")
        raise


if __name__ == "__main__":
    main() 