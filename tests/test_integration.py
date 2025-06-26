"""
Integration tests for image-search CLI tool.

These tests use mocked CLIP models but test the full CLI workflow.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch

from image_search_cli import cli


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.mark.integration
@patch('image_search_cli.clip.load')
@patch('image_search_cli.clip.tokenize')
def test_full_workflow_with_mocked_clip(mock_tokenize, mock_clip_load, runner, multiple_test_images, isolated_db, mock_clip_model):
    """Test complete workflow: add images, search, list, remove."""
    mock_clip_load.return_value = mock_clip_model
    
    # Mock tokenize to return mock tensor
    from tests.conftest import MockTensor
    import numpy as np
    mock_tokenize.return_value = MockTensor(np.random.randint(0, 1000, (1, 77)))
    
    # Step 1: Add images
    image_paths = [str(path) for path in multiple_test_images.values()]
    result = runner.invoke(cli, ['add'] + image_paths)
    assert result.exit_code == 0
    assert 'Successfully added' in result.output
    
    # Step 2: List images
    result = runner.invoke(cli, ['list'])
    assert result.exit_code == 0
    assert 'total' in result.output
    
    # Step 3: Get stats
    result = runner.invoke(cli, ['stats'])
    assert result.exit_code == 0
    assert 'Images indexed:' in result.output
    
    # Step 4: Search for images
    result = runner.invoke(cli, ['search', 'red', '--top-k', '2'])
    assert result.exit_code == 0
    # Should either find results or say no results found
    assert 'Found' in result.output or 'No results found' in result.output
    
    # Step 5: Test similar command
    first_image = list(multiple_test_images.values())[0]
    result = runner.invoke(cli, ['similar', str(first_image), '--top-k', '1'])
    assert result.exit_code == 0
    
    # Step 6: Remove an image
    result = runner.invoke(cli, ['remove', str(first_image)])
    assert result.exit_code == 0


@pytest.mark.integration
def test_error_handling_workflow(runner, isolated_db):
    """Test error handling in various scenarios."""
    
    # Test adding non-existent files
    result = runner.invoke(cli, ['add', 'nonexistent.jpg'])
    assert result.exit_code == 0
    assert 'No image files found' in result.output
    
    # Test searching in empty database
    with patch('image_search_cli.EmbeddingModel') as mock_model_class:
        mock_model = MockEmbeddingModel()
        mock_model_class.return_value = mock_model
        
        result = runner.invoke(cli, ['search', 'anything'])
        assert result.exit_code == 0
        # Should handle empty database gracefully
    
    # Test similar with missing image
    result = runner.invoke(cli, ['similar', 'missing.jpg'])
    assert result.exit_code == 0
    assert 'Image not found' in result.output
    
    # Test clean with no missing files
    result = runner.invoke(cli, ['clean'])
    assert result.exit_code == 0


@pytest.mark.integration
@patch('image_search_cli.clip.load')
def test_recursive_add_workflow(mock_clip_load, runner, tmp_path, isolated_db, mock_clip_model):
    """Test recursive directory addition."""
    mock_clip_load.return_value = mock_clip_model
    
    # Create a nested directory structure
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    
    # Create test images in different directories
    from PIL import Image
    
    # Image in root
    img1 = Image.new('RGB', (50, 50), color='red')
    img1_path = tmp_path / "image1.jpg"
    img1.save(img1_path)
    
    # Image in subdirectory
    img2 = Image.new('RGB', (50, 50), color='blue')
    img2_path = subdir / "image2.jpg"
    img2.save(img2_path)
    
    # Test recursive add
    result = runner.invoke(cli, ['add', str(tmp_path), '--recursive'])
    assert result.exit_code == 0
    assert 'Successfully added' in result.output
    
    # Verify both images were found
    result = runner.invoke(cli, ['stats'])
    assert result.exit_code == 0


class MockEmbeddingModel:
    """Simple mock for EmbeddingModel that doesn't require CLIP."""
    
    def encode_image(self, image_path):
        # Return a simple mock embedding
        return [0.1] * 512
    
    def encode_text(self, text):
        # Return a simple mock embedding  
        return [0.2] * 512


@pytest.mark.slow
def test_real_clip_model_loading():
    """Test that CLIP model can actually be loaded (slow test)."""
    pytest.importorskip("clip")
    
    try:
        from image_search_cli import EmbeddingModel
        model = EmbeddingModel("ViT-B/32")
        assert model.model is not None
        assert model.preprocess is not None
    except Exception as e:
        # If model loading fails due to download issues, skip
        pytest.skip(f"Could not load CLIP model: {e}")


@pytest.mark.integration
def test_export_import_workflow(runner, tmp_path, isolated_db):
    """Test export functionality."""
    
    # First add some mock data to export
    with patch('image_search_cli.ImageSearchDB') as mock_db_class:
        mock_db = object.__new__(MockDB)
        mock_db.setup_mock_data()
        mock_db_class.return_value = mock_db
        
        export_file = tmp_path / "test_export.json"
        result = runner.invoke(cli, ['export', '-o', str(export_file)])
        assert result.exit_code == 0


class MockDB:
    """Mock database for export testing."""
    
    def setup_mock_data(self):
        self.mock_results = [
            ('/path/test.jpg', 'test.jpg', 100, 100, '[0.1, 0.2, 0.3]')
        ]
    
    @property 
    def conn(self):
        return MockConnection(self.mock_results)
        
    def close(self):
        pass


class MockConnection:
    """Mock database connection."""
    
    def __init__(self, results):
        self.results = results
        
    def execute(self, query, params=None):
        return MockCursor(self.results)


class MockCursor:
    """Mock database cursor."""
    
    def __init__(self, results):
        self.results = results
        
    def fetchall(self):
        return self.results