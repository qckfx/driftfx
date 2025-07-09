# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

driftfx is a Python package for zero-false-positive data drift detection in analytics pipelines. It uses DAFSA (Directed Acyclic Finite State Automaton) and BK-tree data structures to efficiently detect anomalies, renames, and new values in categorical data columns.

## Development Commands

### Installation
```bash
# Install in development mode
pip install -e .

# Install for production
pip install .
```

### CLI Usage
```bash
# Create a snapshot
driftfx snapshot --csv data.csv --column country > snapshot.json

# Check for drift
driftfx check --csv new_data.csv --column country --snapshot snapshot.json
```

## Code Architecture

The codebase follows a simple, focused architecture:

1. **driftfx/core.py**: Contains all core functionality
   - `DafsaBkTree`: Custom data structure combining DAFSA and BK-tree for efficient string storage and similarity search
   - `snapshot()`: Creates a baseline from DataFrame categorical columns
   - `check()`: Compares new data against baseline to detect drift

2. **driftfx/__main__.py**: CLI entry point using argparse for command-line interface

3. **Dual Interface Design**: 
   - Programmatic API for Python integration
   - CLI for batch processing and shell scripting

## Key Implementation Details

- Uses edit distance (Levenshtein) for typo/rename detection
- Memory-efficient string storage with DAFSA
- Fast approximate matching with BK-tree structure
- Returns structured results with rename suggestions and new values

## Current Limitations

- No test suite exists yet
- No linting or formatting configuration