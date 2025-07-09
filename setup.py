#!/usr/bin/env python
"""Setup script for guardrail with Cython extensions."""

import os
import sys
from setuptools import setup, Extension, find_packages
from pathlib import Path

# Check if Cython is available
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("Warning: Cython not found. Building without optimized extensions.")

# Define extensions
extensions = []

if USE_CYTHON:
    ext_modules = [
        Extension(
            "guardrail._cython.levenshtein",
            ["guardrail/_cython/levenshtein.pyx"],
            extra_compile_args=["-O3", "-march=native"] if sys.platform != "win32" else ["/O2"],
        )
    ]
    extensions = cythonize(ext_modules, compiler_directives={
        'language_level': '3',
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
    })

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="guardrail",
    version="0.1.0-alpha",
    description="Zero-false-positive data drift guardrails using DAFSA + BK-tree",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chris Wood",
    author_email="chris.wood@qckfx.com",
    url="https://github.com/qckfx/guardrail",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
    ],
    ext_modules=extensions,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Cython",
    ],
    entry_points={
        "console_scripts": [
            "guardrail=guardrail.core:main",
        ],
    },
)