# Data Clustering

This package profiles tabular and multimedia inputs and returns data characteristics with rule-based clustering recommendations. It does not train or execute the recommended algorithms.

## Install

```bash
py -3 -m pip install -e .
```

## Analyze a file

From the repository root, run:

```bash
py -3 -m data_clustering path\to\dataset.csv
```

The command prints a JSON analysis report. The same API can be used from Python:

```python
from data_clustering import analyze_file

report = analyze_file(r"path\to\dataset.csv")
print(report.to_dict())
```

Supported tabular inputs are CSV, TSV, Excel (`.xlsx`, `.xls`, `.xlsm`), JSON, and Parquet. Image, audio, and video files can be analyzed individually or by passing a directory. Unsupported file types are rejected with an explicit error rather than guessed.

The report includes text indicators and keywords, numerical statistics, category cardinality and distribution, confidence indicators and missingness, time-series trend/seasonality/autocorrelation characteristics, multimedia types and available metadata, and algorithm recommendations. For ordinal categories not automatically recognized from an ordered pandas categorical or a common scale, pass an explicit order to `analyze_dataframe`:

```python
from data_clustering import analyze_dataframe

report = analyze_dataframe(
    dataframe,
    ordinal_orders={"rating": ["poor", "fair", "good", "excellent"]},
)
```

Image dimensions are extracted with Pillow. Audio/video stream metadata is extracted when FFmpeg's `ffprobe` is installed. Multimedia clustering requires modality-specific feature extraction; this package recommends approaches but does not create embeddings. Parquet support requires a working PyArrow installation.

## Run tests

```bash
py -3 -m pytest -q
```
