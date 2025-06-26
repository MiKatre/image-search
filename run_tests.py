#!/usr/bin/env python3
"""
Simple test runner for image-search CLI tool.

Usage:
    python run_tests.py          # Run all fast tests
    python run_tests.py --all    # Run all tests including slow ones
    python run_tests.py --unit   # Run only unit tests
    python run_tests.py --cov    # Run tests with coverage
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n🔧 {description}")
    print(f"📝 Running: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
    else:
        print(f"❌ {description} - FAILED")
    
    return result.returncode == 0


def main():
    """Main test runner."""
    args = sys.argv[1:]
    
    print("🧪 Image Search CLI Test Runner")
    print("=" * 60)
    
    # Base pytest command
    base_cmd = ["uv", "run", "pytest", "tests/", "-v"]
    
    success = True
    
    if "--all" in args:
        # Run all tests including slow ones
        cmd = base_cmd + []
        success &= run_command(cmd, "Running ALL tests (including slow)")
        
    elif "--unit" in args:
        # Run only unit tests
        cmd = base_cmd + ["-m", "unit"]
        success &= run_command(cmd, "Running UNIT tests only")
        
    elif "--cov" in args:
        # Run with coverage
        cmd = base_cmd + ["-m", "unit", "--cov=image_search_cli", "--cov-report=term-missing"]
        success &= run_command(cmd, "Running tests WITH COVERAGE")
        
    else:
        # Default: run fast tests (unit tests)
        cmd = base_cmd + ["-m", "unit"]
        success &= run_command(cmd, "Running FAST tests (unit only)")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())