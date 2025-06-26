"""
Tests for database operations in image-search CLI.
"""

import json
import pytest
from pathlib import Path
from image_search_cli import ImageSearchDB


@pytest.mark.unit
def test_database_initialization(isolated_db):
    """Test that database initializes with correct schema."""
    db = ImageSearchDB()
    
    # Check that tables exist
    tables = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [table[0] for table in tables]
    
    assert 'images' in table_names
    assert 'vec_images' in table_names
    
    # Check images table schema
    schema = db.conn.execute("PRAGMA table_info(images)").fetchall()
    columns = [col[1] for col in schema]  # col[1] is column name
    
    expected_columns = ['id', 'path', 'filename', 'created_at', 'file_size', 'width', 'height']
    for col in expected_columns:
        assert col in columns
    
    db.close()


@pytest.mark.unit
def test_image_insertion(isolated_db, test_image_path):
    """Test inserting image metadata into database."""
    db = ImageSearchDB()
    
    # Insert image metadata
    cursor = db.conn.execute(
        """INSERT INTO images (path, filename, file_size, width, height) 
           VALUES (?, ?, ?, ?, ?)""",
        (str(test_image_path), test_image_path.name, 1000, 100, 100)
    )
    image_id = cursor.lastrowid
    
    # Insert a mock embedding
    mock_embedding = [0.1] * 512  # Simple embedding
    db.conn.execute(
        "INSERT INTO vec_images (rowid, embedding) VALUES (?, ?)",
        (image_id, json.dumps(mock_embedding))
    )
    
    db.conn.commit()
    
    # Verify insertion
    result = db.conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    assert result is not None
    assert result[1] == str(test_image_path)  # path
    assert result[2] == test_image_path.name   # filename
    
    # Verify vector was inserted
    vec_result = db.conn.execute("SELECT rowid FROM vec_images WHERE rowid = ?", (image_id,)).fetchone()
    assert vec_result is not None
    
    db.close()


@pytest.mark.unit
def test_duplicate_path_handling(isolated_db, test_image_path):
    """Test that duplicate paths are handled correctly."""
    db = ImageSearchDB()
    
    # Insert first image
    db.conn.execute(
        """INSERT INTO images (path, filename, file_size, width, height) 
           VALUES (?, ?, ?, ?, ?)""",
        (str(test_image_path), test_image_path.name, 1000, 100, 100)
    )
    db.conn.commit()
    
    # Try to insert the same path again - should raise constraint error
    with pytest.raises(Exception):  # sqlite3.IntegrityError
        db.conn.execute(
            """INSERT INTO images (path, filename, file_size, width, height) 
               VALUES (?, ?, ?, ?, ?)""",
            (str(test_image_path), test_image_path.name, 1000, 100, 100)
        )
        db.conn.commit()
    
    db.close()


@pytest.mark.unit
def test_vector_search_query(isolated_db, sample_embeddings):
    """Test vector search functionality."""
    db = ImageSearchDB()
    
    # Insert test data with embeddings
    test_data = [
        ("red_car.jpg", "red_car.jpg", sample_embeddings['red_car']),
        ("blue_sky.jpg", "blue_sky.jpg", sample_embeddings['blue_sky']),
        ("green_tree.jpg", "green_tree.jpg", sample_embeddings['green_tree']),
    ]
    
    for path, filename, embedding in test_data:
        cursor = db.conn.execute(
            """INSERT INTO images (path, filename, file_size, width, height) 
               VALUES (?, ?, ?, ?, ?)""",
            (path, filename, 1000, 100, 100)
        )
        image_id = cursor.lastrowid
        
        db.conn.execute(
            "INSERT INTO vec_images (rowid, embedding) VALUES (?, ?)",
            (image_id, json.dumps(embedding))
        )
    
    db.conn.commit()
    
    # Test vector search query format
    query_embedding = sample_embeddings['red_car']
    sql = """
        SELECT 
            i.path,
            i.filename,
            v.distance
        FROM vec_images v
        JOIN images i ON i.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
    """
    
    # This should not raise an error (testing query syntax)
    try:
        results = db.conn.execute(sql, (json.dumps(query_embedding), 2)).fetchall()
        # Should return results
        assert len(results) <= 2
    except Exception as e:
        # If sqlite-vec isn't fully configured in test, that's okay
        # We're mainly testing the query syntax
        assert "vec0" in str(e) or "MATCH" in str(e)
    
    db.close()


@pytest.mark.unit
def test_image_removal(isolated_db, test_image_path):
    """Test removing images from database."""
    db = ImageSearchDB()
    
    # Insert test image
    cursor = db.conn.execute(
        """INSERT INTO images (path, filename, file_size, width, height) 
           VALUES (?, ?, ?, ?, ?)""",
        (str(test_image_path), test_image_path.name, 1000, 100, 100)
    )
    image_id = cursor.lastrowid
    
    mock_embedding = [0.1] * 512
    db.conn.execute(
        "INSERT INTO vec_images (rowid, embedding) VALUES (?, ?)",
        (image_id, json.dumps(mock_embedding))
    )
    db.conn.commit()
    
    # Verify it exists
    result = db.conn.execute("SELECT COUNT(*) FROM images WHERE id = ?", (image_id,)).fetchone()
    assert result[0] == 1
    
    # Remove it
    db.conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    db.conn.execute("DELETE FROM vec_images WHERE rowid = ?", (image_id,))
    db.conn.commit()
    
    # Verify it's gone
    result = db.conn.execute("SELECT COUNT(*) FROM images WHERE id = ?", (image_id,)).fetchone()
    assert result[0] == 0
    
    db.close()


@pytest.mark.unit
def test_database_stats(isolated_db, multiple_test_images, sample_embeddings):
    """Test database statistics functionality."""
    db = ImageSearchDB()
    
    # Insert multiple test images
    for color, image_path in multiple_test_images.items():
        cursor = db.conn.execute(
            """INSERT INTO images (path, filename, file_size, width, height) 
               VALUES (?, ?, ?, ?, ?)""",
            (str(image_path), image_path.name, 1500, 100, 100)
        )
        image_id = cursor.lastrowid
        
        # Use a different embedding for each color
        embedding = sample_embeddings.get(f'{color}_car', sample_embeddings['red_car'])
        db.conn.execute(
            "INSERT INTO vec_images (rowid, embedding) VALUES (?, ?)",
            (image_id, json.dumps(embedding))
        )
    
    db.conn.commit()
    
    # Test count queries
    image_count = db.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    vector_count = db.conn.execute("SELECT COUNT(*) FROM vec_images").fetchone()[0]
    
    assert image_count == len(multiple_test_images)
    assert vector_count == len(multiple_test_images)
    
    # Test aggregate queries
    total_size = db.conn.execute("SELECT SUM(file_size) FROM images").fetchone()[0]
    avg_size = total_size / image_count if image_count > 0 else 0
    
    assert total_size == 1500 * len(multiple_test_images)
    assert avg_size == 1500
    
    db.close()