"""
Tests for CLI commands in image-search tool.
"""

import json
import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, Mock

from image_search_cli import cli, ImageSearchDB, EmbeddingModel


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_embedding_model():
    """Mock EmbeddingModel for CLI tests."""
    mock_model = Mock(spec=EmbeddingModel)
    mock_model.encode_image.return_value = [0.1] * 512
    mock_model.encode_text.return_value = [0.2] * 512
    return mock_model


@pytest.mark.unit
def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Image similarity search' in result.output
    assert 'Commands:' in result.output
    assert 'add' in result.output
    assert 'search' in result.output


@pytest.mark.unit
def test_cli_invalid_command(runner):
    """Test CLI with invalid command."""
    result = runner.invoke(cli, ['invalid-command'])
    assert result.exit_code != 0
    assert 'No such command' in result.output


@pytest.mark.unit
def test_add_command_help(runner):
    """Test add command help."""
    result = runner.invoke(cli, ['add', '--help'])
    assert result.exit_code == 0
    assert 'Add images to the search index' in result.output
    assert '--recursive' in result.output


@pytest.mark.unit
def test_search_command_help(runner):
    """Test search command help."""
    result = runner.invoke(cli, ['search', '--help'])
    assert result.exit_code == 0
    assert 'Search for images similar to the given text query' in result.output
    assert '--top-k' in result.output
    assert '--threshold' in result.output


@pytest.mark.unit
def test_stats_command_empty_db(runner, isolated_db):
    """Test stats command with empty database."""
    with patch('image_search_cli.ImageSearchDB') as mock_db_class:
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchone.return_value = [0]
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['stats'])
        assert result.exit_code == 0
        assert 'No images indexed yet' in result.output


@pytest.mark.unit 
def test_list_command_empty_db(runner, isolated_db):
    """Test list command with empty database."""
    with patch('image_search_cli.ImageSearchDB') as mock_db_class:
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchall.return_value = []
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['list'])
        assert result.exit_code == 0
        assert 'No images indexed yet' in result.output


@pytest.mark.unit
def test_add_command_missing_files(runner):
    """Test add command with non-existent files."""
    result = runner.invoke(cli, ['add', 'nonexistent.jpg'])
    assert result.exit_code == 0  # Command runs but finds no files
    assert 'No image files found' in result.output


@pytest.mark.unit
def test_search_command_no_query(runner):
    """Test search command without query argument."""
    result = runner.invoke(cli, ['search'])
    assert result.exit_code != 0
    assert 'Missing argument' in result.output


@pytest.mark.unit
def test_similar_command_missing_image(runner):
    """Test similar command with non-existent image."""
    result = runner.invoke(cli, ['similar', 'nonexistent.jpg'])
    assert result.exit_code == 0
    assert 'Image not found' in result.output


@pytest.mark.unit
def test_remove_command_missing_image(runner, isolated_db):
    """Test remove command with non-existent image."""
    with patch('image_search_cli.ImageSearchDB') as mock_db_class:
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchone.return_value = None
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['remove', 'nonexistent.jpg'])
        assert result.exit_code == 0
        assert 'Image not found' in result.output


@pytest.mark.unit
def test_clean_command_no_missing_files(runner, isolated_db):
    """Test clean command when no files are missing."""
    with patch('image_search_cli.ImageSearchDB') as mock_db_class:
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchall.return_value = []
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['clean'])
        assert result.exit_code == 0
        assert 'No missing images found' in result.output


@pytest.mark.unit
def test_export_command_empty_db(runner, isolated_db):
    """Test export command with empty database."""
    with patch('image_search_cli.ImageSearchDB') as mock_db_class:
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchall.return_value = []
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['export'])
        assert result.exit_code == 0
        assert 'No data to export' in result.output


@pytest.mark.unit
@patch('image_search_cli.clip.load')
def test_add_command_with_mock_clip(mock_clip_load, runner, test_image_path, isolated_db, mock_clip_model):
    """Test add command with mocked CLIP model."""
    mock_clip_load.return_value = mock_clip_model
    
    with patch('image_search_cli.ImageSearchDB') as mock_db_class:
        mock_db = Mock()
        mock_db.conn.execute.return_value.lastrowid = 1
        mock_db.conn.execute.return_value.fetchone.return_value = None  # No existing image
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['add', str(test_image_path)])
        assert result.exit_code == 0
        assert 'Successfully added' in result.output


@pytest.mark.unit
@patch('image_search_cli.clip.load')
@patch('image_search_cli.clip.tokenize')
def test_search_command_with_mock_clip(mock_tokenize, mock_clip_load, runner, isolated_db, mock_clip_model):
    """Test search command with mocked CLIP model."""
    mock_clip_load.return_value = mock_clip_model
    mock_tokenize.return_value = Mock()  # Mock tokenized text
    
    with patch('image_search_cli.ImageSearchDB') as mock_db_class:
        mock_db = Mock()
        # Mock search results
        mock_db.conn.execute.return_value.fetchall.return_value = [
            ('/path/to/image.jpg', 'image.jpg', 100, 100, 0.5)
        ]
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['search', 'test query'])
        assert result.exit_code == 0
        assert 'Found' in result.output and 'similar images' in result.output


@pytest.mark.unit
def test_model_option(runner):
    """Test CLI with custom model option."""
    # Test that model option is accepted (even if we can't test loading)
    result = runner.invoke(cli, ['--model', 'ViT-B/16', '--help'])
    assert result.exit_code == 0
    

@pytest.mark.unit
def test_search_with_top_k_option(runner, isolated_db):
    """Test search command with top-k option."""
    with patch('image_search_cli.ImageSearchDB') as mock_db_class, \
         patch('image_search_cli.EmbeddingModel') as mock_model_class:
        
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchall.return_value = []
        mock_db_class.return_value = mock_db
        
        mock_model = Mock()
        mock_model.encode_text.return_value = [0.1] * 512
        mock_model_class.return_value = mock_model
        
        result = runner.invoke(cli, ['search', 'test', '--top-k', '3'])
        assert result.exit_code == 0


@pytest.mark.unit
def test_search_with_threshold_option(runner, isolated_db):
    """Test search command with threshold option."""
    with patch('image_search_cli.ImageSearchDB') as mock_db_class, \
         patch('image_search_cli.EmbeddingModel') as mock_model_class:
        
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchall.return_value = []
        mock_db_class.return_value = mock_db
        
        mock_model = Mock()
        mock_model.encode_text.return_value = [0.1] * 512
        mock_model_class.return_value = mock_model
        
        result = runner.invoke(cli, ['search', 'test', '--threshold', '0.5'])
        assert result.exit_code == 0


@pytest.mark.unit
def test_reset_command_help(runner):
    """Test reset command help."""
    result = runner.invoke(cli, ['reset', '--help'])
    assert result.exit_code == 0
    assert 'Reset the image search database' in result.output
    assert '--confirm' in result.output
    assert '--dry-run' in result.output


@pytest.mark.unit
def test_reset_no_database(runner, tmp_path):
    """Test reset command when no database exists."""
    with patch('image_search_cli.Path.home') as mock_home:
        mock_home.return_value = tmp_path
        result = runner.invoke(cli, ['reset'])
        assert result.exit_code == 0
        assert 'No database found' in result.output


@pytest.mark.unit
def test_reset_dry_run(runner, tmp_path):
    """Test reset command with dry-run option."""
    # Create a fake database file
    db_dir = tmp_path / '.image-search'
    db_dir.mkdir()
    db_file = db_dir / 'database.db'
    db_file.write_text('fake database')
    
    with patch('image_search_cli.Path.home') as mock_home, \
         patch('image_search_cli.ImageSearchDB') as mock_db_class:
        
        mock_home.return_value = tmp_path
        
        # Mock database stats
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchone.return_value = [5]  # 5 images
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['reset', '--dry-run'])
        assert result.exit_code == 0
        assert 'DRY RUN' in result.output
        assert 'Files that would be removed' in result.output
        assert str(db_file) in result.output
        
        # Verify file still exists after dry run
        assert db_file.exists()


@pytest.mark.unit
def test_reset_with_confirm_flag(runner, tmp_path):
    """Test reset command with --confirm flag."""
    # Create a fake database file
    db_dir = tmp_path / '.image-search'
    db_dir.mkdir()
    db_file = db_dir / 'database.db'
    db_file.write_text('fake database')
    
    with patch('image_search_cli.Path.home') as mock_home, \
         patch('image_search_cli.ImageSearchDB') as mock_db_class:
        
        mock_home.return_value = tmp_path
        
        # Mock database stats
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchone.return_value = [3]  # 3 images
        mock_db_class.return_value = mock_db
        
        result = runner.invoke(cli, ['reset', '--confirm'])
        assert result.exit_code == 0
        assert 'Database reset complete' in result.output
        
        # Verify file was deleted
        assert not db_file.exists()


@pytest.mark.unit
def test_reset_cancelled_by_user(runner, tmp_path):
    """Test reset command cancelled by user input."""
    # Create a fake database file
    db_dir = tmp_path / '.image-search'
    db_dir.mkdir()
    db_file = db_dir / 'database.db'
    db_file.write_text('fake database')
    
    with patch('image_search_cli.Path.home') as mock_home, \
         patch('image_search_cli.ImageSearchDB') as mock_db_class:
        
        mock_home.return_value = tmp_path
        
        # Mock database stats
        mock_db = Mock()
        mock_db.conn.execute.return_value.fetchone.return_value = [2]  # 2 images
        mock_db_class.return_value = mock_db
        
        # Simulate user saying 'no'
        result = runner.invoke(cli, ['reset'], input='n\n')
        assert result.exit_code == 0
        assert 'Reset cancelled' in result.output
        
        # Verify file still exists
        assert db_file.exists()