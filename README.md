# Data Clustering

This package profiles tabular and multimedia inputs and returns data characteristics with rule-based clustering recommendations. It does not train or execute the recommended algorithms.

## Install

```bash
py -3 -m pip install -e ".[frontend]"
```

## Start the frontend

From the repository root, run:

```bash
py -3 -m streamlit run app.py
```

The browser UI accepts supported tabular and individual media files, displays detected characteristics and recommendations, and lets you download the report as JSON. Uploaded content is written to a temporary directory for analysis and removed when analysis finishes.

For tables, the UI also shows the first five rows, missing values and duplicate
rows. Select the columns to prepare and click **Preprocess dataset**. The
tabular pipeline removes duplicate rows, imputes median/mode, standardizes
numerical columns and one-hot encodes categorical and boolean columns. You
can inspect the resulting preview and report and download the full result as
UTF-8 CSV. Original data and recommendations remain separate from this output.

Dates, free text and media references are not selected by default. String
columns selected explicitly are encoded as whole categories; this pipeline
does not implement TF-IDF, timestamp conversion or multimedia embeddings.
The prepared table is not a universal input for every recommended algorithm.
Blank strings are reported separately from pandas missing values and are not
automatically imputed. Changing the upload clears the previous session results.

## Dataset information and preprocessing API

The shared classifier preserves numeric-looking strings (such as `"001"`) as
categories when they are still strings in the loaded DataFrame. File readers
can infer numeric types before classification. Boolean columns are identified
separately and encoded with the categorical pipeline. Numerical columns with
low cardinality are reported as such without automatically changing their role.

```python
from data_clustering import get_dataset_info, preprocess_dataset, print_preprocessing_report

info = get_dataset_info(dataframe)
processed, report = preprocess_dataset(dataframe)
print_preprocessing_report(report)
```

The preprocessing function accepts numerical, categorical and boolean columns.
It raises `PreprocessingError` for incompatible data, including unsupported
types or entirely missing columns, and never modifies the input DataFrame.
The result retains the original indices of the surviving rows.

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

To show the original terminal demo and run its unified checks, use:

```bash
py -3.14 tests/test_dataset_processing.py
```

The package must first be installed with the editable installation above.
