# Image Search CLI

A command-line tool for semantic image search using CLIP embeddings and vector similarity. Search your image collections using natural language descriptions or find visually similar images.

## Features

- **🔍 Semantic Search**: Find images using natural language queries like "red car in parking lot" or "sunset over mountains"
- **🖼️ Visual Similarity**: Find images similar to a reference image
- **⚡ Fast Vector Search**: Uses SQLite with sqlite-vec extension for efficient similarity search
- **🧠 CLIP Embeddings**: Powered by OpenAI's CLIP model for multimodal understanding
- **📁 Recursive Processing**: Scan entire directory trees for images
- **💾 Local Database**: All data stored locally at `~/.image-search/database.db`
- **📤 Export Capabilities**: Export embeddings and metadata to JSON

## Installation

### Requirements
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install Globally
```bash
# Clone or download this repository
git clone <repository-url>
cd image-search

# Install globally with uv
uv tool install .
```

The `image-search` command will now be available from anywhere on your system.

### Local Development
```bash
# Install dependencies
uv sync

# Run locally
uv run python ./image-search --help
```

## Quick Start

```bash
# Add images to your search index
image-search add ~/Photos/*.jpg --recursive

# Search using natural language
image-search search "red car" --top-k 5

# Find similar images
image-search similar ~/Photos/vacation.jpg

# View your indexed images
image-search list

# Check database statistics
image-search stats
```

## Commands

### `add` - Index Images
Add images to the search index with support for recursive directory scanning.

```bash
# Add specific images
image-search add photo1.jpg photo2.png

# Add all images from a directory
image-search add ~/Photos/*.jpg

# Recursively scan directories
image-search add ~/Pictures --recursive
```

**Supported formats**: JPG, JPEG, PNG, GIF, BMP, WEBP

### `search` - Text-to-Image Search
Search for images using natural language descriptions.

```bash
# Basic search
image-search search "sunset over ocean"

# Limit results
image-search search "red flowers" --top-k 3

# Set similarity threshold
image-search search "blue sky" --threshold 0.3
```

### `similar` - Image-to-Image Search
Find images visually similar to a reference image.

```bash
# Find similar images
image-search similar reference-photo.jpg

# Limit results
image-search similar reference-photo.jpg --top-k 10
```

### `list` - Show Indexed Images
Display all images in your search index with metadata.

```bash
image-search list
```

### `stats` - Database Statistics
Show information about your image database.

```bash
image-search stats
```

### `remove` - Remove Images
Remove specific images from the index.

```bash
# Remove by filename or path
image-search remove photo.jpg
image-search remove ~/Photos/vacation.jpg
```

### `clean` - Clean Missing Files
Remove database entries for files that no longer exist.

```bash
image-search clean
```

### `reset` - Reset Database
Reset the image search database by deleting all data.

```bash
# Interactive confirmation (safe default)
image-search reset

# Skip confirmation prompt
image-search reset --confirm

# Preview what would be deleted
image-search reset --dry-run
```

**⚠️ Warning**: This permanently deletes all indexed images and embeddings. Use with caution!

### `export` - Export Data
Export embeddings and metadata to JSON format.

```bash
# Export to default filename
image-search export

# Export to specific file
image-search export --output my-embeddings.json
```

## How It Works

1. **Image Processing**: Images are processed using OpenAI's CLIP model (ViT-B/32) to generate 512-dimensional embeddings
2. **Vector Storage**: Embeddings are stored in SQLite with the sqlite-vec extension for efficient similarity search
3. **Semantic Understanding**: CLIP's multimodal training enables matching between text descriptions and image content
4. **Similarity Search**: Uses cosine similarity in the embedding space to find relevant images

## Configuration

### Model Selection
You can specify a different CLIP model using the `--model` option:

```bash
image-search --model ViT-B/16 search "landscape"
```

Available models: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `RN50`, `RN101`, etc.

### Database Location
The database is automatically created at `~/.image-search/database.db`. This location is shared across all invocations of the tool.

## Technical Details

- **CLIP Model**: Uses OpenAI CLIP for multimodal embeddings
- **Vector Database**: SQLite with sqlite-vec extension
- **Embedding Dimension**: 512 (ViT-B/32) or 768 (ViT-L/14)
- **Similarity Metric**: Cosine similarity with normalized vectors
- **Storage Format**: Binary vectors with JSON metadata export

## Troubleshooting

### Model Download
On first run, the CLIP model (~338MB for ViT-B/32) will be downloaded automatically. Ensure you have a stable internet connection.

### Performance
- Indexing speed: ~1-2 images per second (depends on image size and hardware)
- Search speed: Sub-second for databases with thousands of images
- Memory usage: ~500MB during processing (for model and image loading)

### Common Issues

**"Model not found"**: Ensure you're using a valid CLIP model name. Use `image-search --help` to see available options.

**"No images found"**: Check that your file paths are correct and files have supported extensions.

**"Database locked"**: Only one instance can write to the database at a time. Wait for other operations to complete.

## Development

### Project Structure
```
image-search/
├── image-search          # Main CLI script (executable)
├── image_search_cli.py   # Python module for global installation
├── pyproject.toml        # Project configuration
├── CLAUDE.md            # Development guidelines
└── tests/               # Test suite
```

### Dependencies
- **click**: CLI framework
- **sqlite-vec**: Vector similarity search extension
- **openai-clip**: CLIP model implementation
- **pillow**: Image processing
- **torch**: Deep learning framework (v2.2.0 for compatibility)

### Running Tests

```bash
# Install test dependencies
uv sync --group test

# Quick unit tests (fast, uses mocked CLIP)
python run_tests.py

# Run with coverage
python run_tests.py --cov

# Run only unit tests
python run_tests.py --unit

# Or use pytest directly
uv run pytest tests/ -m unit -v
```

The test suite includes:
- **Unit tests**: Fast tests with mocked CLIP model (28 tests)
- **Integration tests**: End-to-end workflow tests with mocked components
- **Database tests**: SQLite operations and schema validation
- **CLI tests**: Command-line interface and argument validation