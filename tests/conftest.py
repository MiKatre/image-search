"""
Shared pytest fixtures for image-search CLI tests.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
from PIL import Image
import sys
import os

# Add the parent directory to the path so we can import our module
sys.path.insert(0, str(Path(__file__).parent.parent))

import image_search_cli


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for isolated testing."""
    db_dir = tmp_path / "test_image_search"
    db_dir.mkdir()
    return db_dir / "test_database.db"


@pytest.fixture
def mock_clip_model():
    """Mock CLIP model that returns fake embeddings."""
    mock_model = Mock()
    mock_preprocess = Mock()
    
    # Mock embedding: return a deterministic vector based on input
    def mock_encode_image(image_tensor):
        # Return a fake embedding that's different for different "images"
        hash_val = hash(str(image_tensor.shape)) % 1000
        embedding = np.random.RandomState(hash_val).normal(0, 1, 512)
        # Normalize like real CLIP embeddings
        embedding = embedding / np.linalg.norm(embedding)
        return MockTensor(embedding)
    
    def mock_encode_text(text_tokens):
        # Return a fake embedding based on text content
        text_str = str(text_tokens)
        hash_val = hash(text_str) % 1000
        embedding = np.random.RandomState(hash_val).normal(0, 1, 512)
        embedding = embedding / np.linalg.norm(embedding)
        return MockTensor(embedding)
    
    mock_model.encode_image = mock_encode_image
    mock_model.encode_text = mock_encode_text
    
    def mock_preprocess_func(image):
        # Return a mock tensor that represents the preprocessed image
        return MockTensor(np.random.rand(3, 224, 224))
    
    mock_preprocess.side_effect = mock_preprocess_func
    
    return mock_model, mock_preprocess


class MockTensor:
    """Mock PyTorch tensor for testing."""
    
    def __init__(self, data):
        self.data = np.array(data)
        
    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self.data, axis=dim))
        
    def to(self, device):
        return self
        
    def cpu(self):
        return self
        
    def numpy(self):
        return self.data
        
    def norm(self, dim=-1, keepdim=True):
        norm_val = np.linalg.norm(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(norm_val)
        
    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data / other.data)
        return MockTensor(self.data / other)
        
    @property
    def shape(self):
        return self.data.shape


@pytest.fixture
def mock_clip_load(mock_clip_model):
    """Mock clip.load function to return our mock model."""
    with patch('image_search_cli.clip.load') as mock_load:
        mock_load.return_value = mock_clip_model
        yield mock_load


@pytest.fixture  
def mock_clip_tokenize():
    """Mock clip.tokenize function."""
    def tokenize_func(texts):
        # Return a mock tensor representing tokenized text
        if isinstance(texts, list):
            return MockTensor(np.random.randint(0, 1000, (len(texts), 77)))
        else:
            return MockTensor(np.random.randint(0, 1000, (1, 77)))
    
    with patch('image_search_cli.clip.tokenize') as mock_tokenize:
        mock_tokenize.side_effect = tokenize_func
        yield mock_tokenize


@pytest.fixture
def test_image():
    """Create a small test image."""
    # Create a simple 100x100 RGB image
    image = Image.new('RGB', (100, 100), color='red')
    return image


@pytest.fixture
def test_image_path(tmp_path, test_image):
    """Create a test image file."""
    image_path = tmp_path / "test_image.jpg"
    test_image.save(image_path)
    return image_path


@pytest.fixture
def multiple_test_images(tmp_path):
    """Create multiple test images with different colors."""
    images = {}
    colors = [('red', (255, 0, 0)), ('blue', (0, 0, 255)), ('green', (0, 255, 0))]
    
    for name, color in colors:
        image = Image.new('RGB', (100, 100), color=color)
        image_path = tmp_path / f"{name}_image.jpg"
        image.save(image_path)
        images[name] = image_path
    
    return images


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing."""
    np.random.seed(42)  # For reproducible test results
    return {
        'red_car': np.random.normal(0, 1, 512).tolist(),
        'blue_sky': np.random.normal(0, 1, 512).tolist(),
        'green_tree': np.random.normal(0, 1, 512).tolist(),
    }


@pytest.fixture
def isolated_db(temp_db_path, monkeypatch):
    """Create an isolated database for testing."""
    # Patch the database path to use our temporary path
    def mock_init(self, db_path=None):
        self.db_path = temp_db_path
        self.db_path.parent.mkdir(exist_ok=True)
        
        import sqlite3
        import sqlite_vec
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self._init_tables()
    
    monkeypatch.setattr(image_search_cli.ImageSearchDB, '__init__', mock_init)
    return temp_db_path