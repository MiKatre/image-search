# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an image search CLI tool that uses CLIP (Contrastive Language-Image Pre-training) embeddings and sqlite-vec for vector similarity search. The tool allows users to search for images using natural language descriptions or find similar images using reference images.

## Architecture

### Core Components

**Main Script**: `image-search` (executable Python script)
- CLI interface built with Click
- Single-file architecture containing all functionality

**Database Layer**: `ImageSearchDB` class
- SQLite database with sqlite-vec extension for vector operations
- Two main tables: `images` (metadata) and `vec_images` (vector embeddings)
- Database stored at `~/.image-search/database.db`

**Embedding Model**: `EmbeddingModel` class  
- Uses OpenAI CLIP library (not sentence-transformers due to PyTorch compatibility issues)
- Default model: ViT-B/32 (512-dimensional embeddings)
- Handles both image and text encoding with proper normalization

### Key Dependencies and Compatibility Notes

**Critical**: This project uses `torch==2.2.0` specifically to avoid CVE-2025-32434 security vulnerability in newer PyTorch versions. The newer versions (2.6+) don't have compatible wheels for macOS x86_64, and sentence-transformers has compatibility issues with PyTorch 2.6+.

**Workaround**: Uses `openai-clip` library instead of `sentence-transformers` to avoid torch.load security issues while maintaining CLIP functionality.

## Development Commands

### Setup and Installation
```bash
# Install dependencies using uv (required)
uv sync

# Install test dependencies
uv sync --group test

# Make script executable
chmod +x image-search

# Test basic functionality
uv run python ./image-search --help
```

### Testing
```bash
# Run fast unit tests (recommended for development)
python run_tests.py

# Run tests with coverage
python run_tests.py --cov

# Run all tests including slow integration tests
python run_tests.py --all

# Run pytest directly with specific markers
uv run pytest tests/ -m unit -v
uv run pytest tests/ -m integration -v
uv run pytest tests/ -m slow -v
```

### Testing Commands
```bash
# Create test images (helper script included)
uv run python create_test_images.py

# Add images to index
uv run python ./image-search add test_images/*.jpg

# Test search functionality
uv run python ./image-search search "red car" -n 3
uv run python ./image-search similar test_images/red_car.jpg

# Database management
uv run python ./image-search stats
uv run python ./image-search list
uv run python ./image-search clean
```

## CLI Commands

- `add <paths>` - Add images to search index (supports --recursive)
- `search <query>` - Text-to-image search with natural language
- `similar <image>` - Find images similar to reference image  
- `list` - Show all indexed images
- `stats` - Display database statistics
- `remove <image>` - Remove image from index
- `clean` - Remove entries for missing files
- `export` - Export embeddings and metadata to JSON

## Important Implementation Details

### Vector Search Query Syntax
sqlite-vec requires specific syntax: `WHERE embedding MATCH ? AND k = ?` (not standard LIMIT clause)

### Similarity Calculation
Normalized CLIP embeddings produce distances in 0-2 range. Similarity percentage calculated as: `max(0, 1 - (distance / 2))`

### Database Schema
- `images` table: metadata (path, filename, dimensions, file_size, created_at)
- `vec_images` table: vector embeddings (FLOAT[512] for ViT-B/32)
- Linked via rowid relationships

### Model Loading
First run downloads ViT-B/32 model (~338MB). Subsequent runs load from cache. Model loads on CPU by default (CUDA if available).

## Known Issues

- setuptools warnings about deprecated pkg_resources (cosmetic, doesn't affect functionality)
- PyTorch compatibility constraints limit upgrade path until ecosystem resolves CVE-2025-32434 properly