# guardrail

🚦  Zero-false-positive drift detection for analytics pipelines.

**Fast**: Optimized with Cython - handles 10,000+ unique values in under 2 seconds.

# Get Started
```bash
pip install guardrail
```

# Python Usage
```python
import pandas as pd, guardrail as gr

gr.snapshot(df_baseline, "baseline", cols=["name"])
result = gr.check(df_new, "baseline", cols=["name"])

if not result.is_clean():
    if result.renames:
        print("Renames / typos:", result.renames[:5])     # first 5 examples
    if result.brand_new:
        print("Brand-new names:", result.brand_new[:5])   # first 5 examples
```

# CLI Usage
```bash
# snapshot baseline
$ guardrail snapshot --input data.parquet --cols name --baseline baseline/
$ Snapshot complete ✓

# check new batch
$ guardrail check --input new.parquet --cols name --baseline baseline/
$ [✖] name: 17 renames / 31 new codes
$ Drift detected: 48 anomalies 
```

# Performance

With Cython-optimized Levenshtein distance calculations:

| Operation | Time | Throughput | Dataset |
|-----------|------|------------|---------|
| Snapshot | 1.9s | 5,264 rows/s | 10,000 unique values |
| Check | 1.3s | 7,893 rows/s | 10,050 rows |

The Cython implementation provides an **8x speedup** for snapshot operations compared to pure Python.