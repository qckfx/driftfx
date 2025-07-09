# guardrail

🚦  Zero-false-positive drift detection for analytics pipelines.

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