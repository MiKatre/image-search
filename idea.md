Here is the idea for a small cli tool I would like to make to easily search for my photos on my computer. You should do it step by step and ask me questions if you need more information, and test each step with simple short tests before moving on to the next one, once the test passes. We should keep it in one script. Keep things simple, do the simple things first.


---

Perfect! Here's the updated CLI structure using sqlite-vec - much cleaner and more powerful:Here's your complete CLI using sqlite-vec! Much cleaner than manual similarity computation.

## Key Features

**Native SQL Vector Search**: Uses sqlite-vec's `MATCH` syntax for fast similarity search
```sql
SELECT * FROM vec_images WHERE embedding MATCH '[query_vector]' ORDER BY distance LIMIT 5
```

**Two Search Modes**:
- `search "sunset over mountains"` - Text-to-image search
- `similar photo.jpg` - Image-to-image search

**Auto-Setup**: Creates database and tables automatically on first run

## Installation & Usage

```bash
# Install dependencies
pip install click requests sqlite-vec pillow

# Make executable
chmod +x image-search

# Add some images
./image-search add ~/photos/*.jpg --recursive

# Search by description
./image-search search "red car in parking lot" --top-k 3

# Find similar images
./image-search similar vacation-photo.jpg

# Management
./image-search list
./image-search stats
./image-search clean  # removes missing files
```

## Benefits Over Manual Approach

- **Much faster**: sqlite-vec uses optimized C code
- **Cleaner code**: No numpy/scipy similarity calculations
- **SQL-native**: All search logic in database
- **Future-proof**: Will support ANN indexes later for even better performance





I think this code could work out of the box but not sure.:
```python
#!/usr/bin/env python3
"""
Image Search CLI using sqlite-vec for vector similarity search.
Uses CLIP for image/text embeddings via sentence-transformers.

Setup:
    uv add click sqlite-vec sentence-transformers pillow torch
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional
import click
import sqlite_vec
from sentence_transformers import SentenceTransformer
from PIL import Image

# Configuration
DEFAULT_MODEL = "clip-ViT-B-32"  # Multimodal CLIP model
EMBEDDING_DIM = 512  # CLIP-ViT-B-32 dimension


class ImageSearchDB:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path.home() / '.image-search' / 'database.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables."""
        # Main images table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                width INTEGER,
                height INTEGER
            )
        ''')
        
        # Vector embeddings table
        self.conn.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_images USING vec0(
                embedding FLOAT[{EMBEDDING_DIM}]
            )
        ''')
        
        self.conn.commit()
    
    def close(self):
        self.conn.close()


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        click.echo(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        click.echo("✓ Model loaded")
    
    def encode_image(self, image_path: str) -> List[float]:
        """Get embedding for an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            embedding = self.model.encode(image, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            click.echo(f"Error encoding image {image_path}: {e}")
            return None
    
    def encode_text(self, text: str) -> List[float]:
        """Get embedding for text (works with CLIP models)."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            click.echo(f"Error encoding text '{text}': {e}")
            return None


@click.group()
@click.option('--model', default=DEFAULT_MODEL, help='Embedding model to use')
@click.pass_context
def cli(ctx, model):
    """Image similarity search using CLIP embeddings and sqlite-vec."""
    ctx.ensure_object(dict)
    ctx.obj['model_name'] = model


@cli.command()
@click.argument('image_paths', nargs=-1, required=True)
@click.option('--recursive', '-r', is_flag=True, help='Recursively add images from directories')
@click.pass_context
def add(ctx, image_paths, recursive):
    """Add images to the search index."""
    db = ImageSearchDB()
    model = EmbeddingModel(ctx.obj['model_name'])
    
    # Collect all image files
    image_files = []
    for path_str in image_paths:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}:
            image_files.append(path)
        elif path.is_dir() and recursive:
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                image_files.extend(path.rglob(f'*{ext}'))
                image_files.extend(path.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        click.echo("No image files found")
        return
    
    click.echo(f"Processing {len(image_files)} images...")
    
    added_count = 0
    for image_path in image_files:
        try:
            # Check if already exists
            existing = db.conn.execute(
                "SELECT id FROM images WHERE path = ?", 
                (str(image_path.absolute()),)
            ).fetchone()
            
            if existing:
                click.echo(f"Skipping {image_path.name} (already indexed)")
                continue
            
            click.echo(f"Processing {image_path.name}...")
            
            # Get image info
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                width, height = None, None
            
            # Get embedding
            embedding = model.encode_image(str(image_path))
            if embedding is None:
                click.echo(f"Failed to get embedding for {image_path.name}")
                continue
            
            # Insert image metadata
            cursor = db.conn.execute(
                """INSERT INTO images (path, filename, file_size, width, height) 
                   VALUES (?, ?, ?, ?, ?)""",
                (str(image_path.absolute()), image_path.name, 
                 image_path.stat().st_size, width, height)
            )
            image_id = cursor.lastrowid
            
            # Insert embedding vector
            db.conn.execute(
                "INSERT INTO vec_images (rowid, embedding) VALUES (?, ?)",
                (image_id, json.dumps(embedding))
            )
            
            db.conn.commit()
            click.echo(f"✓ Added {image_path.name}")
            added_count += 1
            
        except Exception as e:
            click.echo(f"Error processing {image_path.name}: {e}")
    
    click.echo(f"\n✓ Successfully added {added_count} images")
    db.close()


@cli.command()
@click.argument('query')
@click.option('--top-k', '-n', default=5, help='Number of results to return')
@click.option('--threshold', '-t', type=float, help='Minimum similarity threshold (0-1)')
@click.pass_context
def search(ctx, query, top_k, threshold):
    """Search for images similar to the given text query."""
    db = ImageSearchDB()
    model = EmbeddingModel(ctx.obj['model_name'])
    
    click.echo(f"Searching for: '{query}'")
    
    # Get query embedding
    query_embedding = model.encode_text(query)
    if query_embedding is None:
        click.echo("Failed to get query embedding")
        return
    
    # Vector search using sqlite-vec
    sql = """
        SELECT 
            i.path,
            i.filename,
            i.width,
            i.height,
            v.distance
        FROM vec_images v
        JOIN images i ON i.id = v.rowid
        WHERE v.embedding MATCH ?
        ORDER BY v.distance
        LIMIT ?
    """
    
    results = db.conn.execute(sql, (json.dumps(query_embedding), top_k)).fetchall()
    
    if not results:
        click.echo("No results found")
        return
    
    click.echo(f"\nFound {len(results)} similar images:")
    click.echo("-" * 80)
    
    for path, filename, width, height, distance in results:
        # Convert distance to similarity (lower distance = higher similarity)
        similarity = max(0, 1 - distance)
        
        if threshold is None or similarity >= threshold:
            dimensions = f"{width}x{height}" if width and height else "unknown"
            click.echo(f"{filename:<30} | {similarity:5.1%} | {dimensions:<10} | {path}")
    
    db.close()


@cli.command()
@click.argument('image_path')
@click.option('--top-k', '-n', default=5, help='Number of results to return')
@click.pass_context
def similar(ctx, image_path, top_k):
    """Find images similar to the given image."""
    db = ImageSearchDB()
    model = EmbeddingModel(ctx.obj['model_name'])
    
    image_path = Path(image_path)
    if not image_path.exists():
        click.echo(f"Image not found: {image_path}")
        return
    
    click.echo(f"Finding images similar to: {image_path.name}")
    
    # Get embedding for the query image
    query_embedding = model.encode_image(str(image_path))
    if query_embedding is None:
        click.echo("Failed to get image embedding")
        return
    
    # Search for similar images
    sql = """
        SELECT 
            i.path,
            i.filename,
            i.width,
            i.height,
            v.distance
        FROM vec_images v
        JOIN images i ON i.id = v.rowid
        WHERE v.embedding MATCH ?
        ORDER BY v.distance
        LIMIT ?
    """
    
    results = db.conn.execute(sql, (json.dumps(query_embedding), top_k + 1)).fetchall()
    
    # Filter out the query image itself
    results = [r for r in results if Path(r[0]).name != image_path.name][:top_k]
    
    if not results:
        click.echo("No similar images found")
        return
    
    click.echo(f"\nFound {len(results)} similar images:")
    click.echo("-" * 80)
    
    for path, filename, width, height, distance in results:
        similarity = max(0, 1 - distance)
        dimensions = f"{width}x{height}" if width and height else "unknown"
        click.echo(f"{filename:<30} | {similarity:5.1%} | {dimensions:<10} | {path}")
    
    db.close()


@cli.command()
def list():
    """List all indexed images."""
    db = ImageSearchDB()
    
    results = db.conn.execute(
        """SELECT filename, path, width, height, created_at 
           FROM images ORDER BY created_at DESC"""
    ).fetchall()
    
    if not results:
        click.echo("No images indexed yet")
        return
    
    click.echo(f"Indexed images ({len(results)} total):")
    click.echo("-" * 90)
    
    for filename, path, width, height, created_at in results:
        dimensions = f"{width}x{height}" if width and height else "unknown"
        click.echo(f"{filename:<30} | {dimensions:<10} | {created_at} | {path}")
    
    db.close()


@cli.command()
@click.argument('image_path')
def remove(image_path):
    """Remove an image from the index."""
    db = ImageSearchDB()
    
    # Find image by path or filename
    image_path = Path(image_path)
    
    # Try absolute path first, then just filename
    result = db.conn.execute(
        "SELECT id, filename FROM images WHERE path = ? OR filename = ?",
        (str(image_path.absolute()), image_path.name)
    ).fetchone()
    
    if not result:
        click.echo(f"Image not found: {image_path.name}")
        return
    
    image_id, filename = result
    
    # Remove from both tables
    db.conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    db.conn.execute("DELETE FROM vec_images WHERE rowid = ?", (image_id,))
    db.conn.commit()
    
    click.echo(f"✓ Removed {filename}")
    db.close()


@cli.command()
def stats():
    """Show database statistics."""
    db = ImageSearchDB()
    
    image_count = db.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    vector_count = db.conn.execute("SELECT COUNT(*) FROM vec_images").fetchone()[0]
    
    if image_count > 0:
        total_size = db.conn.execute("SELECT SUM(file_size) FROM images").fetchone()[0] or 0
        avg_size = total_size / image_count if image_count > 0 else 0
        
        # Get some sample dimensions
        sample_dims = db.conn.execute(
            "SELECT width, height FROM images WHERE width IS NOT NULL LIMIT 5"
        ).fetchall()
        
        click.echo(f"Database Statistics:")
        click.echo(f"Images indexed: {image_count}")
        click.echo(f"Vectors stored: {vector_count}")
        click.echo(f"Total file size: {total_size / 1024 / 1024:.1f} MB")
        click.echo(f"Average file size: {avg_size / 1024:.1f} KB")
        click.echo(f"Embedding dimension: {EMBEDDING_DIM}")
        
        if sample_dims:
            click.echo(f"Sample dimensions: {', '.join(f'{w}x{h}' for w, h in sample_dims[:3])}")
    else:
        click.echo("No images indexed yet")
    
    db.close()


@cli.command()
def clean():
    """Remove entries for missing image files."""
    db = ImageSearchDB()
    
    results = db.conn.execute("SELECT id, path, filename FROM images").fetchall()
    removed = 0
    
    for image_id, path, filename in results:
        if not Path(path).exists():
            db.conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
            db.conn.execute("DELETE FROM vec_images WHERE rowid = ?", (image_id,))
            click.echo(f"Removed missing: {filename}")
            removed += 1
    
    if removed > 0:
        db.conn.commit()
        click.echo(f"\n✓ Removed {removed} missing images")
    else:
        click.echo("No missing images found")
    
    db.close()


@cli.command()
@click.option('--output', '-o', help='Output file for embeddings export')
def export(output):
    """Export embeddings and metadata to JSON."""
    db = ImageSearchDB()
    
    if output is None:
        output = f"image_embeddings_{Path.cwd().name}.json"
    
    # Get all data
    results = db.conn.execute("""
        SELECT i.path, i.filename, i.width, i.height, v.embedding
        FROM images i
        JOIN vec_images v ON i.id = v.rowid
        ORDER BY i.filename
    """).fetchall()
    
    if not results:
        click.echo("No data to export")
        return
    
    export_data = {
        'model': DEFAULT_MODEL,
        'embedding_dim': EMBEDDING_DIM,
        'images': []
    }
    
    for path, filename, width, height, embedding_json in results:
        export_data['images'].append({
            'path': path,
            'filename': filename,
            'width': width,
            'height': height,
            'embedding': json.loads(embedding_json)
        })
    
    with open(output, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    click.echo(f"✓ Exported {len(results)} embeddings to {output}")
    db.close()


if __name__ == '__main__':
    cli()
```