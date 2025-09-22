-- Additional MySQL-specific indexes and optimizations
-- Run this after the main schema for performance optimization

USE bookdb;

-- Composite indexes for common query patterns
CREATE INDEX idx_user_interactions_lookup ON user_interactions (user_id, timestamp DESC, action);
CREATE INDEX idx_book_interactions_lookup ON user_interactions (book_id, timestamp DESC, action);
CREATE INDEX idx_recent_interactions ON user_interactions (timestamp DESC, action, book_id);

-- Indexes for recommendation queries
CREATE INDEX idx_books_recommendation ON books (genre, rating DESC, popularity_score DESC);
CREATE INDEX idx_books_popularity_rating ON books (popularity_score DESC, rating DESC);

-- Partitioning for large user_interactions table (optional)
-- This example partitions by year - adjust based on your data volume
-- ALTER TABLE user_interactions 
-- PARTITION BY RANGE (YEAR(timestamp)) (
--     PARTITION p2023 VALUES LESS THAN (2024),
--     PARTITION p2024 VALUES LESS THAN (2025),
--     PARTITION p2025 VALUES LESS THAN (2026),
--     PARTITION p_future VALUES LESS THAN MAXVALUE
-- );

-- Full-text search indexes for book content
ALTER TABLE books ADD FULLTEXT(title, author, description);

-- Stored procedures for common operations
DELIMITER //

-- Get user's recent interaction history
CREATE PROCEDURE GetUserHistory(IN p_user_id VARCHAR(50), IN p_limit INT)
BEGIN
    SELECT book_id, action, dwell_time, timestamp
    FROM user_interactions 
    WHERE user_id = p_user_id 
    ORDER BY timestamp DESC 
    LIMIT p_limit;
END //

-- Get popular books in the last N days
CREATE PROCEDURE GetPopularBooks(IN p_days INT, IN p_limit INT)
BEGIN
    SELECT 
        ui.book_id,
        COUNT(*) as interaction_count,
        AVG(ui.dwell_time) as avg_dwell_time,
        b.title,
        b.author,
        b.rating
    FROM user_interactions ui
    JOIN books b ON ui.book_id = b.book_id
    WHERE ui.timestamp >= DATE_SUB(NOW(), INTERVAL p_days DAY)
        AND ui.action IN ('view', 'borrow', 'click')
    GROUP BY ui.book_id, b.title, b.author, b.rating
    ORDER BY interaction_count DESC
    LIMIT p_limit;
END //

-- Get books by genre with popularity
CREATE PROCEDURE GetBooksByGenre(IN p_genre VARCHAR(100), IN p_limit INT)
BEGIN
    SELECT book_id, title, author, rating, popularity_score
    FROM books 
    WHERE genre = p_genre 
    ORDER BY popularity_score DESC, rating DESC
    LIMIT p_limit;
END //

-- Update book popularity score based on recent interactions
CREATE PROCEDURE UpdateBookPopularity()
BEGIN
    -- Update popularity scores based on last 30 days interactions
    UPDATE books b
    SET popularity_score = LEAST(1.0, (
        SELECT COALESCE(
            (COUNT(*) * 0.1 + AVG(dwell_time) * 0.001) / 10, 
            0.0
        )
        FROM user_interactions ui
        WHERE ui.book_id = b.book_id
            AND ui.timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            AND ui.action IN ('view', 'borrow', 'click')
    ));
END //

DELIMITER ;

-- Create events for automatic maintenance (optional)
-- SET GLOBAL event_scheduler = ON;

-- CREATE EVENT update_book_popularity
-- ON SCHEDULE EVERY 1 DAY
-- STARTS CURRENT_TIMESTAMP
-- DO
--   CALL UpdateBookPopularity();

-- Performance monitoring views
CREATE OR REPLACE VIEW interaction_performance AS
SELECT 
    DATE(timestamp) as interaction_date,
    COUNT(*) as total_interactions,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT book_id) as unique_books,
    AVG(dwell_time) as avg_dwell_time
FROM user_interactions
WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(timestamp)
ORDER BY interaction_date DESC;

CREATE OR REPLACE VIEW book_performance AS
SELECT 
    b.book_id,
    b.title,
    b.genre,
    COUNT(ui.interaction_id) as total_interactions,
    COUNT(DISTINCT ui.user_id) as unique_users,
    AVG(ui.dwell_time) as avg_dwell_time,
    b.rating,
    b.popularity_score
FROM books b
LEFT JOIN user_interactions ui ON b.book_id = ui.book_id
    AND ui.timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY b.book_id, b.title, b.genre, b.rating, b.popularity_score
ORDER BY total_interactions DESC;

-- Query optimization hints for MySQL
-- Add these as comments for common queries:

-- For user history queries:
-- SELECT /*+ USE_INDEX(user_interactions, idx_user_interactions_lookup) */ ...

-- For popular books queries:
-- SELECT /*+ USE_INDEX(user_interactions, idx_recent_interactions) */ ...

COMMIT;
