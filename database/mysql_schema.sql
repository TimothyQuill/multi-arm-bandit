-- MySQL Schema for Book Recommendation System
-- Run this to create the necessary tables for the recommendation engine

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS bookdb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE bookdb;

-- Books table
CREATE TABLE IF NOT EXISTS books (
    book_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    author VARCHAR(200),
    genre VARCHAR(100),
    publication_year INT,
    page_count INT,
    rating DECIMAL(3,2) DEFAULT 0.00,
    popularity_score DECIMAL(5,4) DEFAULT 0.0000,
    availability DECIMAL(3,2) DEFAULT 1.00,
    isbn VARCHAR(20),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_genre (genre),
    INDEX idx_rating (rating),
    INDEX idx_popularity (popularity_score),
    INDEX idx_publication_year (publication_year)
) ENGINE=InnoDB;

-- User interactions table
CREATE TABLE IF NOT EXISTS user_interactions (
    interaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    book_id VARCHAR(50) NOT NULL,
    action ENUM('view', 'click', 'borrow', 'return', 'favorite', 'ignore') NOT NULL,
    dwell_time DECIMAL(8,2) DEFAULT 0.00,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(100),
    confidence_score DECIMAL(5,4),
    INDEX idx_user_id (user_id),
    INDEX idx_book_id (book_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_action (action),
    INDEX idx_user_book (user_id, book_id),
    INDEX idx_user_timestamp (user_id, timestamp DESC),
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Users table (simplified - only essential demographic information)
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    age INT,
    gender ENUM('male', 'female', 'other', 'unknown') DEFAULT 'unknown',
    library_branch VARCHAR(50) DEFAULT 'unknown',
    occupation_category ENUM('student', 'professional', 'retired', 'unemployed', 'other', 'unknown') DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_age (age),
    INDEX idx_library_branch (library_branch),
    INDEX idx_occupation (occupation_category),
    INDEX idx_gender (gender)
) ENGINE=InnoDB;

-- Library branches table
CREATE TABLE IF NOT EXISTS library_branches (
    branch_id VARCHAR(50) PRIMARY KEY,
    branch_name VARCHAR(200) NOT NULL,
    address TEXT,
    contact_phone VARCHAR(20),
    contact_email VARCHAR(100),
    opening_hours JSON,
    branch_type ENUM('main_library', 'branch', 'mobile', 'digital') DEFAULT 'branch',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_branch_type (branch_type),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB;

-- Insert sample library branches
INSERT INTO library_branches (branch_id, branch_name, branch_type) VALUES
('main_library', 'Main Library', 'main_library'),
('central_branch', 'Central Branch', 'branch'),
('branch_a', 'Branch A - North District', 'branch'),
('branch_b', 'Branch B - South District', 'branch'),
('branch_c', 'Branch C - East District', 'branch'),
('branch_d', 'Branch D - West District', 'branch'),
('branch_e', 'Branch E - University Campus', 'branch'),
('branch_f', 'Branch F - Shopping Centre', 'branch'),
('branch_g', 'Branch G - Community Centre', 'branch'),
('branch_h', 'Branch H - School District', 'branch'),
('branch_i', 'Branch I - Business District', 'branch'),
('branch_j', 'Branch J - Residential Area', 'branch'),
('branch_k', 'Branch K - Suburban', 'branch'),
('branch_l', 'Branch L - Historic District', 'branch'),
('branch_m', 'Branch M - Waterfront', 'branch'),
('branch_n', 'Branch N - Industrial', 'branch'),
('branch_o', 'Branch O - Cultural Quarter', 'branch')
ON DUPLICATE KEY UPDATE branch_name = VALUES(branch_name);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    duration_minutes INT DEFAULT 0,
    device ENUM('mobile', 'tablet', 'desktop') DEFAULT 'desktop',
    location VARCHAR(100),
    time_of_day ENUM('morning', 'afternoon', 'evening', 'night'),
    INDEX idx_user_id (user_id),
    INDEX idx_start_time (start_time),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Book embeddings metadata table (optional - for tracking embedding versions)
CREATE TABLE IF NOT EXISTS book_embeddings (
    book_id VARCHAR(50) PRIMARY KEY,
    collaborative_embedding_version INT DEFAULT 1,
    content_embedding_version INT DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Performance optimization indexes for recommendation queries
CREATE INDEX idx_interactions_recent ON user_interactions (timestamp DESC, book_id);
CREATE INDEX idx_interactions_user_recent ON user_interactions (user_id, timestamp DESC);
CREATE INDEX idx_books_popularity_genre ON books (genre, popularity_score DESC);

-- Views for common queries
CREATE OR REPLACE VIEW popular_books_last_30_days AS
SELECT 
    book_id,
    COUNT(*) as interaction_count,
    AVG(dwell_time) as avg_dwell_time
FROM user_interactions 
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    AND action IN ('view', 'borrow', 'click')
GROUP BY book_id
ORDER BY interaction_count DESC;

CREATE OR REPLACE VIEW user_interaction_summary AS
SELECT 
    user_id,
    COUNT(*) as total_interactions,
    COUNT(DISTINCT book_id) as unique_books,
    AVG(dwell_time) as avg_dwell_time,
    MAX(timestamp) as last_interaction
FROM user_interactions
GROUP BY user_id;

-- Sample data insertion (optional)
-- INSERT INTO books (book_id, title, author, genre, publication_year, page_count, rating, popularity_score)
-- VALUES 
--     ('book_001', 'The Great Gatsby', 'F. Scott Fitzgerald', 'fiction', 1925, 180, 4.2, 0.85),
--     ('book_002', 'To Kill a Mockingbird', 'Harper Lee', 'fiction', 1960, 281, 4.5, 0.92),
--     ('book_003', '1984', 'George Orwell', 'scifi', 1949, 328, 4.6, 0.95);

COMMIT;
