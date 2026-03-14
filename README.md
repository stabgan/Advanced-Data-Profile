# Advanced Data Profile

Automated CSV data profiling tool that generates comprehensive, interactive HTML reports with statistical analysis, visualizations, and NLP insights.

## What It Does

Drop CSV files into a `data/` folder and run the profiler. For each file it produces a self-contained, compressed HTML report covering:

- **Table-level overview** — row/column counts, memory usage, duplicate detection, data volume
- **Column-level analysis** — data type inference (integer, float, string, date, timestamp), null/not-null counts, uniqueness index, top values
- **Statistical profiling** — descriptive stats, skewness, kurtosis, outlier detection (z-score), median, percentiles
- **Visualizations** — histograms, box plots, Q-Q plots, cumulative frequency plots, correlation heatmaps, PCA feature importance
- **Text analysis** — TF-IDF n-gram scoring, word clouds, Flesch readability scores, character/syllable counts
- **Language detection** — identifies non-English text using FastText with confidence scores
- **Date/time intelligence** — automatic format detection across hundreds of permutations with a persistent format cache
- **Categorical detection** — entropy-based confidence scoring for categorical columns

Reports are minified (HTML/CSS/JS) and gzip-compressed for efficient storage.

## Project Structure

```
├── main.py                        # Entry point — iterates CSVs in data/ and runs all phases
├── Data_Profiler_TCS/
│   ├── Profiler.py                # Core profiling engine (DataProfile class, 5 phases + report gen)
│   ├── jinja_template.html        # Bootstrap-based HTML report template
│   └── data/
│       ├── lid.176.ftz            # FastText language identification model
│       └── input_data.csv         # Sample input data
├── output/                        # Generated reports land here
├── requirements.txt               # Python dependencies
└── .gitignore
```

## 🛠 Tech Stack

| | Technology | Purpose |
|---|---|---|
| 🐼 | `pandas` / `numpy` / `scipy` | Data manipulation and statistics |
| 📊 | `plotly` / `matplotlib` / `datashader` | Charting and visualization |
| 🤖 | `scikit-learn` | PCA, TF-IDF, imputation, scaling |
| 🗣️ | `spacy` + `en_core_web_sm` | NLP tokenization and POS tagging |
| 🌐 | `fasttext` | Language detection |
| 📝 | `textstat` / `wordcloud` | Readability metrics and word clouds |
| 📅 | `dateparser` | Fallback date parsing |
| 🧩 | `jinja2` | HTML report templating |
| 🗜️ | `beautifulsoup4` / `htmlmin` / `cssmin` / `jsmin` | Report minification |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

1. Place CSV files in a `data/` directory at the project root.
2. Run the profiler:

```bash
python main.py
```

3. Find compressed HTML reports in `output/` (`.html.gz` files).

You can also use the `DataProfile` class directly:

```python
from Data_Profiler_TCS import Profiler

dp = Profiler.DataProfile(
    file_path="data/my_data.csv",
    env_name="prod",
    schema_name="my_schema",
    table_name="my_table",
    output_folder="output",
    skip_col_stats="N",
    skip_table_stats="N",
    sample_size_for_plots=5000,
)

dp.first_phase()
dp.second_phase()
dp.third_phase()
dp.fourth_phase()
dp.fifth_phase()
dp.generate_html_report()
```

## ⚠️ Known Issues

- **Large binary files in repo** — `venv.zip` (118 MB) and FastText model files are tracked in git history; consider using Git LFS
- **No tests** — there are no unit or integration tests
- **Global side effects at import** — `Profiler.py` loads spaCy and FastText models at import time, which slows startup
- **Commented-out code** — significant blocks of dead code (KMeans clustering, POS tagging charts) remain in `Profiler.py`
