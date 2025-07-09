# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

driftfx is a Python package for zero-false-positive data drift detection in analytics pipelines. It uses DAFSA (Directed Acyclic Finite State Automaton) and BK-tree data structures to efficiently detect anomalies, renames, and new values in categorical data columns.

## Development Commands

### Installation
```bash
# Install in development mode with Cython
pip install "Cython>=3.0"
pip install -e .

# Build Cython extensions in-place
python setup.py build_ext --inplace

# Install without Cython (pure Python fallback)
pip install . --no-build-isolation

# Build wheel distribution
python -m build
```

### CLI Usage
```bash
# Create a snapshot
driftfx snapshot --input data.csv --cols country city --baseline ./baseline

# Check for drift
driftfx check --input new_data.csv --cols country city --baseline ./baseline

# Also works with Parquet files
driftfx snapshot --input data.parquet --cols country --baseline ./baseline
```

## Code Architecture

The codebase follows a simple, focused architecture:

1. **driftfx/core.py**: Contains all core functionality (424 lines)
   - `DafsaBkTree`: Custom data structure combining DAFSA and BK-tree for efficient string storage and similarity search
   - `snapshot()`: Creates a baseline from DataFrame categorical columns
   - `check()`: Compares new data against baseline to detect drift
   - CLI argument parsing and main entry point

2. **driftfx/_cython/levenshtein.pyx**: Cython-optimized Levenshtein distance implementation
   - 8x performance improvement over pure Python
   - Graceful fallback if Cython unavailable
   - Compiler optimizations: `-O3 -march=native` (Unix), `/O2` (Windows)

3. **Build System**:
   - Modern PEP 517/518 packaging with `pyproject.toml`
   - Optional Cython dependency with runtime fallback
   - Entry point: `driftfx` command → `driftfx.core:main`

## Key Implementation Details

- Uses edit distance (Levenshtein) for typo/rename detection
- Memory-efficient string storage with DAFSA
- Fast approximate matching with BK-tree structure
- Returns structured results with rename suggestions and new values
- Supports both CSV and Parquet file formats via pandas
- Parallel processing for large anomaly sets (>50 items)

## Current Limitations

- No test suite exists yet
- No linting or formatting configuration
- No CI/CD pipeline
- No type annotations or mypy configuration