# 📊 Advanced Data Profiler

Automated CSV data profiling engine that generates comprehensive, self-contained HTML reports packed with statistical analysis, interactive visualizations, NLP insights, and language detection.

Drop your CSVs in, get rich profiling reports out — compressed and ready to share.

---

## 🔍 What It Does

For each CSV file, the profiler runs five analysis phases and produces a minified, gzip-compressed HTML report covering:

- **Table overview** — row/column counts, memory usage, duplicate detection, data volume
- **Column-level analysis** — data type inference (integer, float, string, date, timestamp), null/not-null counts, uniqueness index, top values
- **Statistical profiling** — descriptive stats, skewness, kurtosis, outlier detection (z-score), median, percentiles
- **Interactive visualizations** — histograms, box plots, Q-Q plots, cumulative frequency plots, correlation heatmaps, PCA feature importance
- **Text / NLP analysis** — TF-IDF n-gram scoring, word clouds, Flesch readability scores, character/syllable counts
- **Language detection** — identifies non-English text via FastText with confidence scores
- **Date/time intelligence** — automatic format detection across hundreds of format permutations with a persistent format cache
- **Categorical detection** — entropy-based confidence scoring for categorical columns

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| 🐍 Language | Python 3.9+ |
| 📦 Data | `pandas` · `numpy` · `scipy` |
| 📈 Visualization | `plotly` · `matplotlib` · `datashader` |
| 🤖 Machine Learning | `scikit-learn` (PCA, TF-IDF, imputation, scaling) |
| 🗣️ NLP | `spacy` (+ `en_core_web_sm`) · `fasttext` · `textstat` |
| 🌐 Language Detection | `fasttext` · `pycountry` |
| ☁️ Word Clouds | `wordcloud` |
| 📅 Date Parsing | `dateparser` |
| 🧾 Templating | `jinja2` |
| 🗜️ Minification | `beautifulsoup4` · `htmlmin` · `cssmin` · `jsmin` |
| 💻 System | `psutil` |

---

## 📁 Project Structure

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

---

## 📦 Dependencies

Install all required packages:

```bash
pip install pandas numpy scipy plotly matplotlib datashader scikit-learn \
  spacy fasttext pycountry textstat wordcloud dateparser jinja2 \
  beautifulsoup4 htmlmin cssmin jsmin psutil
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 How to Run

### Quick Start

1. Place your `.csv` files in a `data/` directory at the project root.
2. Run the profiler:

```bash
python main.py
```

3. Find compressed HTML reports in the `output/` folder (`.html.gz` files).

### Programmatic Usage

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

dp.first_phase()           # Load data, infer types, detect duplicates
dp.second_phase()          # Aggregate type counts, null summaries
dp.third_phase()           # Per-column detailed profiling
dp.fourth_phase()          # Statistical analysis + visualizations
dp.fifth_phase()           # Correlation heatmap + PCA
dp.generate_html_report()  # Render, minify, compress → .html.gz
```

---

## ⚠️ Known Issues

- **No `requirements.txt`** — dependencies must be installed manually (see above).
- **No `.gitignore`** — IDE files, `__pycache__/`, `*.pkl`, and `venv.zip` are tracked.
- **`venv.zip` checked in** — a ~118 MB virtual environment archive is in the repo.
- **Hardcoded paths** — `main.py` expects a `data/` folder at the project root; `format_cache.pkl` writes to CWD.
- **Global side effects at import** — spaCy model, FastText model, and format cache load on import.
- **No tests** — no unit or integration tests exist.

---

## 📄 License

No license file is included in this repository.
