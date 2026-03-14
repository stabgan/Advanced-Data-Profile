# Advanced Data Profile

Automated CSV data profiling tool that generates comprehensive, interactive HTML reports with statistical analysis, visualizations, and NLP insights.

## What It Does

Drop CSV files into a `data/` folder and run the profiler. For each file, it produces a self-contained compressed HTML report covering:

- **Table-level overview** — row/column counts, memory usage, duplicate detection, data volume
- **Column-level analysis** — data type inference (integer, float, string, date, timestamp), null/not-null counts, uniqueness index, top values
- **Statistical profiling** — descriptive stats, skewness, kurtosis, outlier detection (z-score), median, percentiles
- **Visualizations** — histograms, box plots, Q-Q plots, cumulative frequency plots, correlation heatmaps, PCA feature importance bar charts
- **Text analysis** — TF-IDF n-gram scoring, word clouds, Flesch readability scores, character/syllable counts
- **Language detection** — identifies non-English text using FastText with confidence scores
- **Date/time intelligence** — automatic format detection across hundreds of format permutations, with a persistent format cache for performance
- **Categorical detection** — entropy-based confidence scoring for categorical columns

Reports are minified (HTML/CSS/JS) and gzip-compressed for efficient storage.

## Project Structure

```
├── main.py                          # Entry point — iterates CSVs in data/ and runs all phases
├── Data_Profiler_TCS/
│   ├── Profiler.py                  # Core profiling engine (DataProfile class, 5 phases + report gen)
│   ├── jinja_template.html          # Bootstrap-based HTML report template
│   └── data/
│       ├── lid.176.ftz              # FastText language identification model
│       └── input_data.csv           # Sample input data
├── output/                          # Generated reports land here
├── format_cache.pkl                 # Cached date format frequencies (auto-generated)
└── qodana.yaml                      # JetBrains Qodana static analysis config
```

## Dependencies

Python 3.x with the following packages:

| Package | Purpose |
|---|---|
| `pandas`, `numpy`, `scipy` | Data manipulation and statistics |
| `plotly`, `matplotlib`, `datashader` | Charting and visualization |
| `scikit-learn` | PCA, TF-IDF, imputation, scaling |
| `spacy` (+ `en_core_web_sm`) | NLP tokenization and POS tagging |
| `fasttext` | Language detection |
| `pycountry` | Language code resolution |
| `textstat` | Readability metrics |
| `wordcloud` | Word cloud generation |
| `dateparser` | Fallback date parsing |
| `jinja2` | HTML report templating |
| `beautifulsoup4`, `htmlmin`, `cssmin`, `jsmin` | Report minification |
| `psutil` | System memory info |

### Setup

```bash
# Install dependencies (no requirements.txt provided — see Known Issues)
pip install pandas numpy scipy plotly matplotlib datashader scikit-learn \
  spacy fasttext pycountry textstat wordcloud dateparser jinja2 \
  beautifulsoup4 htmlmin cssmin jsmin psutil

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

## Usage

1. Place your CSV files in a `data/` directory at the project root.
2. Run the profiler:

```bash
python main.py
```

3. Find compressed HTML reports in the `output/` folder (`.html.gz` files).

You can also use the `DataProfile` class directly:

```python
from Data_Profiler_TCS import Profiler

dp = Profiler.DataProfile(
    file_path="data/my_data.csv",
    env_name="prod",
    schema_name="my_schema",
    table_name="my_table",
    output_folder="output",
    skip_col_stats="N",      # "Y" to skip column-level stats
    skip_table_stats="N",    # "Y" to skip table-level stats (PCA, correlation)
    sample_size_for_plots=5000  # Optional: cap sample size for large datasets
)

dp.first_phase()    # Load data, infer types, detect duplicates
dp.second_phase()   # Aggregate type counts, null summaries
dp.third_phase()    # Per-column detailed profiling
dp.fourth_phase()   # Statistical analysis + visualizations
dp.fifth_phase()    # Correlation heatmap + PCA
dp.generate_html_report()  # Render, minify, compress → .html.gz
```

## Known Issues

- **No `requirements.txt` or `pyproject.toml`** — dependencies must be installed manually.
- **No `.gitignore`** — `.idea/`, `__pycache__/`, `*.pkl`, `venv.zip`, and the PDF file are committed to the repo.
- **`venv.zip` checked in** — a 118 MB virtual environment archive is tracked in the repository.
- **Hardcoded paths** — `main.py` expects a `data/` folder at the project root; `format_cache.pkl` is written to the current working directory.
- **Global side effects at import time** — `Profiler.py` loads the spaCy model, FastText model, and prints the format cache on import, which slows startup and pollutes stdout.
- **Date validation range** — hardcoded to 1900–2030, which will need updating.
- **No tests** — there are no unit or integration tests.
- **Commented-out code** — significant blocks of dead code (KMeans clustering, POS tagging charts) remain in `Profiler.py`.

## License

No license file is included in this repository.
