# Standard library imports
import contextlib
import json
import re
from datetime import datetime
from io import BytesIO
from typing import Union, Any
import functools
import pandas
import pycountry
import unicodedata
from sklearn.decomposition import PCA
import dateparser
import fasttext
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly.graph_objs import Histogram, Box, Scatter, Figure, Pie, Bar
import psutil
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import entropy, skew
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import textstat
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
import spacy
import gzip
import htmlmin
from jsmin import jsmin
from cssmin import cssmin
from importlib import resources
from bs4 import BeautifulSoup
import pickle
import os
from itertools import product
import datashader as ds
from colorcet import fire

import io
import base64
from datashader import reductions as rd
import datashader.transfer_functions as tf


# Define your minification functions here
def minify_js(js_content):
    return jsmin(js_content)


def minify_css(css_content):
    return cssmin(css_content)


def minify_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Minify inline JavaScript
    for script in soup.find_all('script'):
        if script.string:
            script.string = minify_js(script.string)

    # Minify inline CSS
    for style in soup.find_all('style'):
        if style.string:
            style.string = minify_css(style.string)

    # Minify the entire HTML after processing inline JS and CSS
    return htmlmin.minify(str(soup), remove_comments=True, remove_empty_space=True)


def compress_html(html_content, output_filepath):
    with gzip.open(output_filepath + '.gz', 'wt', encoding='utf-8') as f:
        f.write(html_content)


nlp = spacy.load('en_core_web_sm')
# Path to the current script (profiler.py)
current_file_path = os.path.abspath(__file__)

# Directory of the current script
current_directory = os.path.dirname(current_file_path)

# Path to the fasttext model relative to the current script
model_path = os.path.join(current_directory, 'data', 'lid.176.bin')

# Load the model

# Load the fasttext model (make sure to provide the correct path to the model file)
with contextlib.redirect_stderr(io.StringIO()):
    ft_model = fasttext.load_model(model_path)
pd.set_option('display.max_columns', None)
# Compile the regex pattern outside of the function
non_word_pattern = re.compile(r'\W+')

# Date Components
years = ['%Y', '%y']
months = ['%m', '%b', '%B']
days = ['%d', '%j']
date_separators = ['/', '-', '.', ' ']

# Time Components
hours = ['%H', '%I']
minutes = ['%M']
seconds = ['%S']
fractions = ['', '%f']
am_pm = ['', '%p']
time_separators = [':', '']
timezones = ['', '%Z', '%z']

# Generate permutations for date and time
date_time_formats = set()
for y, m, d, dss in product(years, months, days, date_separators):
    date_format = f"{y}{ds}{m}{dss}{d}".strip()
    for h, mins, s, fs, ampm, ts1, ts2, tz in product(hours, minutes, seconds, fractions, am_pm, time_separators,
                                                      time_separators, timezones):
        time_format = f"{h}{ts1}{mins}{ts2}{s}{fs}{ampm}{tz}".strip()
        # Combine date and time formats
        combined_format = f"{date_format} {time_format}".strip()
        date_time_formats.add(combined_format)

# Convert the set to a list and sort it
date_time_formats_list = sorted(list(date_time_formats))

# Global cache for successful formats
format_cache_file = 'format_cache.pkl'
format_cache = {}

# Load format cache if exists
if os.path.exists(format_cache_file):
    with open(format_cache_file, 'rb') as f:
        format_cache = pickle.load(f)


def validate_date(dates):
    """Validate the parsed dates."""
    year = dates.dt.year
    month = dates.dt.month
    day = dates.dt.day
    return ((1900 <= year) & (year <= 2030) & (1 <= month) & (month <= 12) & (1 <= day) & (day <= 31)).all()


def parse_dates_with_format(column, fmt):
    """Parse dates using a specific format."""
    try:
        parsed_dates = pd.to_datetime(column, format=fmt, errors='raise')
        if validate_date(parsed_dates):
            return parsed_dates
    except (ValueError, TypeError):
        return None


def try_parse_date(column, date_time_formats_list):
    """Try parsing dates using various methods."""
    # Check cache first
    cached_format = format_cache.get(column.name)
    if cached_format:
        parsed_dates = parse_dates_with_format(column, cached_format)
        if parsed_dates is not None:
            return parsed_dates

    # Common formats
    common_date_formats_list = [
        '%Y-%m-%d %H:%M:%S.%f',  # Example: 2023-12-22 04:22:30.615016
        '%Y-%m-%d %H:%M:%S',  # Example: 2023-12-22 04:22:30
        '%Y-%m-%d',  # Example: 2023-12-22
        '%H:%M:%S',  # Example: 04:22:30
        '%Y/%m/%d %H:%M:%S',  # Example: 2023/12/22 04:22:30
        '%Y/%m/%d',  # Example: 2023/12/22
        '%m/%d/%Y %H:%M:%S',  # Example: 12/22/2023 04:22:30 (American style)
        '%m/%d/%Y',  # Example: 12/22/2023 (American style)
        '%d/%m/%Y %H:%M:%S',  # Example: 22/12/2023 04:22:30 (European style)
        '%d/%m/%Y',  # Example: 22/12/2023 (European style)
        '%Y %B %d %H:%M:%S',  # Example: 2023 December 22 04:22:30
        '%Y %B %d',  # Example: 2023 December 22
        '%Y %b %d %H:%M:%S',  # Example: 2023 Dec 22 04:22:30
        '%Y %b %d',  # Example: 2023 Dec 22
        '%Y %m %d %H:%M:%S',  # Example: 2023 12 22 04:22:30
        '%Y %m %d'  # Example: 2023 12 22
    ]
    for fmt in common_date_formats_list:
        parsed_dates = parse_dates_with_format(column, fmt)
        if parsed_dates is not None:
            format_cache[column.name] = fmt  # Update cache
            return parsed_dates

    # Sequential processing for the remaining formats
    for fmt in date_time_formats_list:
        result = parse_dates_with_format(column, fmt)
        if result is not None:
            format_cache[column.name] = fmt  # Update cache
            return result

    # Fallback to default parsing
    return pd.to_datetime(column, errors='coerce')


def custom_data_type(col):
    """Determine the custom data type of a column."""
    if col.isnull().all():
        return 'empty'

    if pd.api.types.is_numeric_dtype(col):
        if (col.dropna() % 1 == 0).all():
            return 'integer'
        return 'float'

    if pd.api.types.is_string_dtype(col):
        parsed_dates = try_parse_date(col.dropna(), date_time_formats_list)
        if not parsed_dates.isnull().all():
            is_date = ((parsed_dates.dt.hour == 0) & (parsed_dates.dt.minute == 0) & (
                    parsed_dates.dt.second == 0)).all()
            return 'date' if is_date else 'timestamp'

    return 'string'


# Custom function to determine the data type of a column


def contains_non_english_characters(col_data):
    # Vectorized operation for non-ASCII character detection
    return col_data.str.contains(r'[^\x00-\x7F]', na=False).any()


@functools.lru_cache(maxsize=128)
def detect_language_with_confidence(text):
    # Ensure that text is a string
    if not isinstance(text, str):
        return [('English', 100)]

    # Check for non-Latin characters using unicodedata
    if any(unicodedata.category(char).startswith('L') and not unicodedata.name(char).startswith('LATIN') for char in
           text):
        # Clean the text data and perform language detection
        cleaned_text = re.sub(r'\W+', ' ', text)
        predictions = ft_model.predict(cleaned_text, k=1)
        languages = predictions[0]
        probabilities = predictions[1]
        full_language_names = []

        for lang_code, prob in zip(languages, probabilities):
            lang_code = lang_code.replace('__label__', '')
            try:
                language = pycountry.languages.get(alpha_2=lang_code)
                full_language_names.append((language.name, round(prob * 100, 2)))
            except AttributeError:
                # If the language code is not found in pycountry, return the code
                full_language_names.append((lang_code, round(prob * 100, 2)))

        return full_language_names
    else:
        # If text contains only Latin characters, assume it's English
        return [('English', 100)]


def calculate_entropy(column: pd.Series) -> float:
    value_counts = column.value_counts()
    probabilities = value_counts / len(column)
    return entropy(probabilities)


def categorical_confidence(col_data: pd.Series, total_rows: int) -> float:
    unique_values = col_data.nunique()
    unique_ratio = unique_values / total_rows
    if unique_values <= 1:  # Single or no value
        return 0.0
    elif unique_ratio < 0.1:
        if pd.api.types.is_float_dtype(col_data):
            col_data = col_data.round(2).astype(str)
        else:
            col_data = col_data.astype(str)
        ent = calculate_entropy(col_data)
        # High entropy -> lower confidence, and vice versa
        confidence = max(0, 100 - ent * 10)
        return round(confidence, 2)
    return 0.0


def is_date_or_timestamp(s):
    if not isinstance(s, str):
        return None
    try:
        parsed_date = dateparser.parse(s)
        if parsed_date:
            return 'date' if parsed_date.hour == 0 and parsed_date.minute == 0 and parsed_date.second == 0 else 'timestamp'
    except Exception as e:
        print(f"Error parsing date: {e}")
    return None


def identify_column_type(col):
    # Apply the is_date_or_timestamp function to each element in the column
    results = col.apply(is_date_or_timestamp)

    # Get unique results
    unique_results = results.unique()

    # If all elements are identified as 'date', return 'date column'
    if len(unique_results) == 1 and unique_results[0] == 'date':
        return 'date column'
    # If all elements are identified as 'timestamp', return 'timestamp column'
    elif len(unique_results) == 1 and unique_results[0] == 'timestamp':
        return 'timestamp column'

    # Function to create histogram


def create_histogram(data, title, xaxis_title, yaxis_title):
    fig = go.Figure(data=[go.Histogram(x=data)])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


# Function to create bar chart
def create_bar_chart(data, title, xaxis_title, yaxis_title):
    # Assumes data is a list of tuples (name, score)
    fig = Figure(data=[Bar(x=[x[0] for x in data], y=[x[1] for x in data])])
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


# Function to convert matplotlib figure to Plotly
def clean_text(text):
    """ Remove non-alphabetic characters and extra spaces. """
    return re.sub(r'[^a-zA-Z\s]', '', text).strip()


def is_valid_text(text, min_word_count=5):
    """ Check if the text has the minimum number of words. """
    return len(text.split()) >= min_word_count  # More efficient word count


# Additional function to convert images to data URI
def img_to_data_uri(fig):
    """Convert matplotlib figure to data URI for Plotly"""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return 'data:image/png;base64,' + img_data


def round_if_float(value):
    # This function will round the value if it is a float
    if isinstance(value, float):
        return round(value, 2)
    return value


def calculate_outlier_percentage(data):
    z_scores = np.abs(stats.zscore(data))
    outliers = data[z_scores > 3]
    return len(outliers) / len(data) * 100


# # List of distributions to check
# distributions = [stats.norm, stats.lognorm, stats.expon, stats.weibull_min, stats.gamma]
#
# def fit_distribution(data, distribution):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         try:
#             params = distribution.fit(data)
#             # Kolmogorov-Smirnov test
#             D, p = stats.kstest(data, distribution.name, args=params)
#             return (distribution.name, D, p, params)
#         except Exception as e:
#             return (distribution.name, None, None, None)  # Indicate failed fit
#
# def best_fit_distribution(data, num_processes=None):
#     if num_processes is None:
#         num_processes = multiprocessing.cpu_count()
#
#     pool = multiprocessing.Pool(processes=num_processes)
#     results = pool.starmap(fit_distribution, [(data, dist) for dist in distributions])
#     pool.close()
#     pool.join()
#
#     # Filter out failed fits and sort by the p-value
#     valid_results = [result for result in results if result[1] is not None]
#     best_fit = sorted(valid_results, key=lambda x: x[1], reverse=True)[0] if valid_results else None
#
#     return best_fit


def fig_to_json(fig):
    """
    Converts a Plotly figure to JSON.
    """
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def get_confidence_color(confidence):
    # Assuming confidence is a percentage
    if confidence >= 85:
        return '#dc3545'  # Red for high confidence
    elif confidence >= 70:
        return '#007bff'  # Blue for medium confidence
    else:
        return '#28a745'  # Green for low confidence


def convert_to_serializable(val):
    """Convert numpy types, nested collections, and Timestamp objects to native Python types for JSON serialization."""

    if isinstance(val, np.generic):
        if np.issubdtype(val, np.bool_):
            return bool(val.item())  # Convert numpy bool_ to Python bool
        return val.item()
    elif isinstance(val, pd.Timestamp):
        return val.isoformat()  # Convert Timestamp to ISO 8601 string
    elif isinstance(val, (dict, list, tuple)):
        if isinstance(val, dict):
            return {k: convert_to_serializable(v) for k, v in val.items()}
        else:  # For lists and tuples
            return type(val)(convert_to_serializable(v) for v in val)
    else:  # No need for conversion if it's already a native Python type
        return val


# Assuming `data` is a list of text documents
def tokenize_and_pos_tag(doc):
    spacy_doc = nlp(doc)
    return [(token.text, token.tag_) for token in spacy_doc]


def parallel_nltk_processing(documents):
    results = [tokenize_and_pos_tag(doc) for doc in documents]
    return results


class DataProfile:
    def __init__(self, file_path, env_name='dummy', schema_name='dummy_schema', table_name='dummy_table',
                 output_folder='.', skip_col_stats='N', skip_table_stats='N', sample_size_for_plots=None,
                 n_cluster_for_kmeans=3):
        self.n_cluster_for_kmeans = n_cluster_for_kmeans
        self.sample_size = sample_size_for_plots
        self.skip_col_stats = skip_col_stats
        self.skip_table_stats = skip_table_stats
        self.jinja_data = None
        self.phase_5_dict = None
        self.phase_4_dict = None
        self.phase_3_list = None
        self.env_name = env_name
        self.schema_name = schema_name
        self.table_name = table_name
        self.file_path = file_path
        self.phase_1_dict = None
        self.random_rows_df = None
        self.df = None
        self.custom_data_types = None
        self.phase_2_dict = None
        self.categorical_info = {}
        # self.pos_tag_descriptions = {
        #     'CC': 'Coordinating conjunction',
        #     'CD': 'Cardinal number',
        #     'DT': 'Determiner',
        #     'EX': 'Existential there',
        #     'FW': 'Foreign word',
        #     'IN': 'Preposition or subordinating conjunction',
        #     'JJ': 'Adjective',
        #     'JJR': 'Adjective, comparative',
        #     'JJS': 'Adjective, superlative',
        #     'LS': 'List item marker',
        #     'MD': 'Modal',
        #     'NN': 'Noun, singular or mass',
        #     'NNS': 'Noun, plural',
        #     'NNP': 'Proper noun, singular',
        #     'NNPS': 'Proper noun, plural',
        #     'PDT': 'Predeterminer',
        #     'POS': 'Possessive ending',
        #     'PRP': 'Personal pronoun',
        #     'PRP$': 'Possessive pronoun',
        #     'RB': 'Adverb',
        #     'RBR': 'Adverb, comparative',
        #     'RBS': 'Adverb, superlative',
        #     'RP': 'Particle',
        #     'SYM': 'Symbol',
        #     'TO': 'to',
        #     'UH': 'Interjection',
        #     'VB': 'Verb, base form',
        #     'VBD': 'Verb, past tense',
        #     'VBG': 'Verb, gerund or present participle',
        #     'VBN': 'Verb, past participle',
        #     'VBP': 'Verb, non-3rd person singular present',
        #     'VBZ': 'Verb, 3rd person singular present',
        #     'WDT': 'Wh-determiner',
        #     'WP': 'Wh-pronoun',
        #     'WP$': 'Possessive wh-pronoun',
        #     'WRB': 'Wh-adverb'}
        # Set up Jinja2 environment
        self.env = Environment(loader=FileSystemLoader('.'))
        with resources.path('Data_Profiler_TCS', 'jinja_template.html') as template_path:
            # Now, set the loader with the directory of the template
            template_directory = template_path.parent
            self.env = Environment(loader=FileSystemLoader(str(template_directory)))
            self.template_name = 'jinja_template.html'
        self.output_folder = output_folder
        self.output_file_name = os.path.join(self.output_folder,
                                             'data_profiling_report_' + self.schema_name + "_" + self.table_name + '_' + datetime.now().strftime(
                                                 "%d%m%Y_%H_%M_%S") + '.html')
        self.custom_data_types = {}

    # Custom function to determine the data type of a column

    def first_phase(self):
        date_time: str = datetime.now().strftime("%m/%d/%Y %I:%M %p")
        data_volume: float = os.path.getsize(self.file_path) / 1024
        total_ram: Union[float, Any] = psutil.virtual_memory().total / (1024 ** 3)
        available_ram: Union[float, Any] = psutil.virtual_memory().available / (1024 ** 3)
        chunksize = 10 ** 6 if data_volume > available_ram else None
        if chunksize:
            df_chunks = pd.read_csv(self.file_path, chunksize=chunksize, low_memory=False)
            self.df = pd.concat(df_chunks, ignore_index=True)
        else:
            self.df = pd.read_csv(self.file_path, low_memory=False)
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
        for column in self.df.columns:
            # Determine the custom data type
            col_type = custom_data_type(self.df[column])

            # Convert to integer if identified as a float but effectively an integer
            if col_type == 'float' and (self.df[column].dropna() % 1 == 0).all():
                self.df[column] = pd.to_numeric(self.df[column], downcast='integer')
                col_type = custom_data_type(self.df[column])  # Recheck the type after conversion

            self.custom_data_types[column] = col_type

        # shape = self.df.shape
        # column_names = self.df.columns.tolist()
        has_duplicates = self.df.duplicated().any()
        n_rows = min(100, self.df.shape[0])
        self.random_rows_df = self.df.sample(n=n_rows)

        # Transpose the DataFrame
        # random_rows_transposed = random_rows_df.T
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024
        self.phase_1_dict = {
            'Environment': self.env_name,
            'Schema Name': self.schema_name,
            'Table Name': self.table_name,
            'Date/Time': date_time,
            'Has Duplicates': 'Yes' if has_duplicates else 'No',
            'Memory Usage in server': f'{memory_usage:.1f} KB',
            'Data volume': f'{data_volume:.1f} KB',
            'Total RAM in server': f'{total_ram:.1f} GB',
            'Available RAM in server': f'{available_ram:.1f} GB',
            'Total Row Count': self.df.shape[0],
            'Total Column Count': self.df.shape[1]
        }

    def second_phase(self):
        # Initialize counters and lists to hold column names for each data type
        int_columns = []
        float_columns = []
        string_columns = []
        date_columns = []
        timestamp_columns = []
        double_columns = []
        max_string_length = 0
        max_decimal_places = 0
        total_zero_percent_count = 0
        total_hundred_percent_count = 0
        total_rows = self.df.shape[0]

        for col in self.df.columns:
            col_data = self.df[col].dropna()
            data_type = self.custom_data_types[col]
            self.categorical_info[col] = categorical_confidence(col_data, total_rows)
            if data_type == 'integer':
                int_columns.append(col)
            elif data_type == 'float':
                float_columns.append(col)
                decimals = col_data.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0)
                max_decimal_places_col = decimals.max()
                max_decimal_places = max(max_decimal_places, max_decimal_places_col)
                if max_decimal_places_col > 6:
                    double_columns.append(col)
            elif data_type == 'string':
                string_columns.append(col)
                max_length_in_col = col_data.astype(str).map(len).max()
                max_string_length = max(max_string_length, max_length_in_col)
                # if categorical_confidence(col_data, total_rows) > 50:
                #     categorical_info[col] = categorical_confidence(col_data, total_rows)
            elif data_type == 'date':
                date_columns.append(col)
            elif data_type == 'timestamp':
                timestamp_columns.append(col)
            if col_data.empty:
                total_zero_percent_count += 1
            elif col_data.count() == total_rows:
                total_hundred_percent_count += 1

        total_null_count = self.df.isnull().sum().sum()
        total_not_null_count = self.df.notnull().sum().sum()
        # print(custom_data_types)

        self.phase_2_dict = {
            # 'Total Data Type Count': unique_data_types_count,
            'Int Column Count': len(int_columns),
            'Float Column Count': len(float_columns),
            'String Column Count': len(string_columns),
            'Date Column Count': len(date_columns),
            'Timestamp Column Count': len(timestamp_columns),
            'Double Column Count': len(double_columns),
            'Maximum String Length': max_string_length,
            'Maximum Decimal Places': max_decimal_places,
            'Total Null Record Count': total_null_count,
            'Total Not Null Record Count': total_not_null_count,
            'Count of columns with no data': total_zero_percent_count,
            'Count of Columns with 100% data': total_hundred_percent_count
        }

    def third_phase(self):
        self.phase_3_list = []

        for col in self.df.columns:
            col_data = self.df[col]
            data_type = self.custom_data_types[col]
            total_count = len(col_data)
            null_count = col_data.isnull().sum()
            non_null_count = total_count - null_count
            unique_values = col_data.nunique()
            top_10_values = col_data.value_counts().head(10).to_dict()

            column_info = {
                "Column Name": col,
                "Column Data Type": data_type,
                "NULL Record Count": null_count,
                "Percentage of NULL Values": round(null_count / total_count * 100, 2),
                "NOT NULL Record Count": non_null_count,
                "Percentage of NOT NULL Values": round(non_null_count / total_count * 100, 2),
                "Number of Distinct Values": unique_values,
                "Uniqueness Index (unique/total)": round(unique_values / total_count * 100, 2),
                "Top 10 Values": top_10_values
            }

            # Additional metrics for numeric and string data
            if data_type == 'integer' or data_type == 'float':
                column_info["Median"] = round(col_data.median(), 2)
                if data_type == 'float':
                    max_decimal_places = col_data.dropna().apply(
                        lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()
                    column_info["Max Decimal Precision"] = max_decimal_places

            if data_type == 'string':
                column_info["Max String Length"] = col_data.astype(str).map(len).max()
                # non_english_chars = any(re.search(r'[^\x00-\x7F]+', str(x)) for x in col_data)
                non_english_chars = contains_non_english_characters(col_data)
                column_info["Contains Non-English Characters"] = contains_non_english_characters(col_data)

                if non_english_chars:
                    unique_texts = col_data.dropna().unique().tolist()
                    languages_with_confidence = [detect_language_with_confidence(text) for text in unique_texts]
                    # Flatten the list of lists to a single list of tuples
                    languages_with_confidence = [item for sublist in languages_with_confidence for item in sublist]

                    # Sort the list of tuples
                    column_info["Languages Detected with Confidence"] = sorted(
                        {x[0]: x for x in languages_with_confidence}.values(), key=lambda x: x[1], reverse=True)[:5]
                else:
                    column_info["Languages Detected with Confidence"] = [("English", 100)]

            if "date" in data_type or "timestamp" in data_type:
                col_data = pd.to_datetime(col_data, errors='coerce')
                column_info["Min Date/Time Value"] = col_data.min()
                column_info["Max Date/Time Value"] = col_data.max()

            self.phase_3_list.append(column_info)

    def use_plotly_for_plots(self, data, col, column_info):
        # Preparing data for plots
        fig_hist = Figure(data=[Histogram(x=data)])
        fig_hist.update_layout(title=f"Histogram of {col}", xaxis_title=col, yaxis_title="Count")

        fig_box = Figure(data=[Box(y=data)])
        fig_box.update_layout(title=f"Box Plot of {col}", yaxis_title=col)

        # Standardize the data
        qq_data = stats.probplot(data, dist="norm")
        theoretical_quantiles = qq_data[0][0]
        ordered_values = qq_data[0][1]

        # Create scatter plot for the Q-Q plot data
        qq_scatter = go.Scatter(x=theoretical_quantiles, y=ordered_values, mode='markers', name='Data')

        # Create a line representing the ideal normal distribution
        line = go.Scatter(x=theoretical_quantiles, y=qq_data[1][1] + qq_data[1][0] * theoretical_quantiles,
                          mode='lines', name='Normal Distribution')

        # Create the figure and add traces
        fig = go.Figure()
        fig.add_trace(qq_scatter)
        fig.add_trace(line)

        # Update the layout
        fig.update_layout(title=f'Q-Q Plot of {col}',
                          xaxis_title='Theoretical Quantiles',
                          yaxis_title='Ordered Values')

        sorted_data = np.sort(data)
        cum_freq = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fig_cum_freq = Figure(data=[Scatter(x=sorted_data, y=cum_freq)])
        fig_cum_freq.update_layout(title=f"Cumulative Frequency Plot of {col}", xaxis_title=col,
                                   yaxis_title="Cumulative Frequency")

        # Converting Plotly figures to JSON
        column_info['histogram'] = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)
        column_info['boxplot'] = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)
        column_info['qq_plot'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        column_info['cumulative_freq'] = json.dumps(fig_cum_freq, cls=plotly.utils.PlotlyJSONEncoder)

    def fourth_phase(self):
        if self.skip_col_stats == 'Y':
            return {}
        self.phase_4_dict = {}

        # Downsize numerical columns and convert to float32 or int32
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = self.df[col].astype('float32')
        for col in self.df.select_dtypes(include=['int64']).columns:
            self.df[col] = self.df[col].astype('int32')

        for col in self.df.columns:
            # Use sampled data for large datasets
            if self.sample_size and len(self.df) > self.sample_size:
                sampled_data = self.df.sample(n=self.sample_size, random_state=42)
                data = sampled_data[col].dropna()
            else:
                data = self.df[col].dropna()

            dtype = self.custom_data_types[col]
            column_info = {'description': {}}
            # Calculating descriptive statistics
            descriptive_stats = data.describe()
            # Mapping the original descriptive statistics keys to more meaningful names
            meaningful_keys = {
                'count': 'Total Count',
                'mean': 'Average',
                'std': 'Standard Deviation',
                'min': 'Minimum Value',
                '25%': '25th Percentile',
                '50%': 'Median (50th Percentile)',
                '75%': '75th Percentile',
                'max': 'Maximum Value'
            }
            # Updating the description dictionary with more meaningful keys
            column_info['description'] = {
                meaningful_keys.get(k, k): round_if_float(v) for k, v in descriptive_stats.to_dict().items()
            }

            if dtype in ['integer', 'float']:
                # Basic statistical analysis
                skewness = round_if_float(data.skew())
                kurtosis = round_if_float(data.kurtosis())
                outlier_percentage = round_if_float(calculate_outlier_percentage(data))

                column_info['description'].update({
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'outlier %': outlier_percentage
                })
                self.use_plotly_for_plots(data, col, column_info)


            elif dtype == 'date' or dtype == 'timestamp':
                try:
                    date_data = pd.to_datetime(data, errors='coerce')
                    # Creating histograms for different time aspects
                    fig_year = create_histogram(date_data.dt.year, f"Year Distribution in {col}", "Year", "Count")
                    fig_month = create_histogram(date_data.dt.month, f"Month Distribution in {col}", "Month", "Count")
                    fig_day = create_histogram(date_data.dt.day, f"Day Distribution in {col}", "Day", "Count")
                    fig_hour = create_histogram(date_data.dt.hour, f"Hour Distribution in {col}", "Hour", "Count")
                    fig_minute = create_histogram(date_data.dt.minute, f"Minute Distribution in {col}", "Minute",
                                                  "Count")
                    fig_second = create_histogram(date_data.dt.second, f"Second Distribution in {col}", "Second",
                                                  "Count")
                    # Converting Plotly figures to JSON
                    column_info['year_hist'] = json.dumps(fig_year, cls=plotly.utils.PlotlyJSONEncoder)
                    column_info['month_hist'] = json.dumps(fig_month, cls=plotly.utils.PlotlyJSONEncoder)
                    column_info['day_hist'] = json.dumps(fig_day, cls=plotly.utils.PlotlyJSONEncoder)
                    column_info['hour_hist'] = json.dumps(fig_hour, cls=plotly.utils.PlotlyJSONEncoder)
                    column_info['minute_hist'] = json.dumps(fig_minute, cls=plotly.utils.PlotlyJSONEncoder)
                    column_info['second_hist'] = json.dumps(fig_second, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(col, e)

            elif dtype == 'string':
                try:
                    # TF-IDF scoring for n-grams
                    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
                    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    tfidf_scores = np.mean(tfidf_matrix, axis=0).tolist()[0]
                    top_ngrams = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:20]
                    fig_tfidf = Figure([Bar(x=[x[0] for x in top_ngrams], y=[x[1] for x in top_ngrams])])
                    fig_tfidf.update_layout(title=f"Top TF-IDF Scores for {col}", xaxis_title="N-grams",
                                            yaxis_title="TF-IDF Score")
                    column_info['tfidf_bar_chart'] = json.dumps(fig_tfidf, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(col, e)
                try:
                    # Generate a word cloud image
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data))
                    plt.figure(figsize=(20, 10), facecolor=None)
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    fig_wordcloud = plt.gcf()
                    data_uri = img_to_data_uri(fig_wordcloud)
                    plt.close()

                    # Create a Plotly figure with the image
                    fig = go.Figure()
                    # Add the word cloud image as a layout image
                    fig.add_layout_image(
                        dict(
                            source=data_uri,
                            xref="x",
                            yref="y",
                            x=0,
                            y=1,
                            sizex=1,
                            sizey=1,
                            sizing="stretch",
                            opacity=1.0,
                            layer="below"
                        )
                    )
                    # Update the layout of the figure to ensure the image fits well
                    fig.update_layout(
                        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0),
                        width=800,  # Adjust the width to fit the modal or container size
                        height=400  # Adjust the height to fit the modal or container size
                    )
                    column_info['word_cloud'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(col, e)

                # Modify the text analysis section
                try:
                    # Text analysis
                    full_text = ' '.join(data).lower()
                    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', full_text).strip()  # Efficient cleaning
                    column_info['description']['flesch reading score'] = round(np.mean(
                        [textstat.flesch_reading_ease(text) if is_valid_text(text) else 0 for text in data]), 2)
                    column_info['description']['character count'] = textstat.char_count(cleaned_text,
                                                                                        ignore_spaces=True)
                    column_info['description']['polysyllable count'] = textstat.polysyllabcount(cleaned_text)
                    column_info['description']['monosyllable count'] = textstat.monosyllabcount(cleaned_text)

                except Exception as e:
                    print(col, e)
                # try:
                #     # Assuming `data` is a list of strings (documents) from your dataframe column
                #     tagged_docs = parallel_nltk_processing(data)
                #
                #     # Aggregate the POS tags from all documents
                #     aggregated_tags = [tag for doc in tagged_docs for _, tag in doc]
                #     pos_counts = Counter(aggregated_tags)
                #
                #     # Prepare x and y data for the bar chart
                #     x_data = list(pos_counts.keys())
                #     y_data = list(pos_counts.values())
                #     hover_text = [f"{get_pos_tag_description(tag, 'Symbols')}: {pos_counts[tag]:,}" for tag in
                #                   pos_counts]
                #
                #     # Create the bar chart
                #     fig = go.Figure(data=[go.Bar(x=x_data, y=y_data, text=hover_text, hoverinfo='text')])
                #
                #     # Update layout to ensure x-axis labels (POS tags) are shown
                #     fig.update_layout(
                #         title=f"Parts of Speech Tag Counts for {col}",
                #         xaxis=dict(
                #             title="POS Tags",
                #             tickmode='array',
                #             tickvals=x_data,
                #             ticktext=x_data,
                #             tickangle=45,  # Rotate the labels to prevent overlap
                #             showticklabels=True  # Ensure that labels are shown
                #         ),
                #         yaxis=dict(
                #             title="Count"
                #         ),
                #         margin=dict(  # Add margins to the layout to ensure labels are not cut off
                #             l=50,
                #             r=50,
                #             b=100,  # Increase bottom margin to accommodate rotated labels
                #             t=50,
                #             pad=4
                #         ),
                #         autosize=False,  # Disable autosize to maintain the set figure size
                #         height=600,  # Set figure height to ensure labels fit
                #         width=800  # Set figure width as necessary
                #     )
                #
                #     # Serialize the POS counts bar chart to JSON
                #     column_info['pos_counts_bar_chart'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                #
                # except Exception as e:
                #     print(col, e)
            self.phase_4_dict[col] = column_info

    # def use_plotly_for_kmeans(self, scaled_data, sampled_df, numerical_cols):
    #     # Assuming scaled_data is your pre-processed data
    #     # Perform PCA with a dynamic number of components based on the data
    #     n_samples, n_features = scaled_data.shape
    #     n_components = min(10, n_samples, n_features)
    #
    #     pca = PCA(n_components=n_components)
    #     pca_result = pca.fit_transform(scaled_data)
    #
    #     # Check if enough samples are available for KMeans
    #     if n_samples < self.n_cluster_for_kmeans:
    #         raise ValueError("Not enough samples for the number of KMeans clusters.")
    #
    #     kmeans = MiniBatchKMeans(n_clusters=self.n_cluster_for_kmeans, random_state=42)
    #     cluster_labels = kmeans.fit_predict(pca_result)
    #
    #     # Generate hover text for each point in the scatter plot
    #     hover_texts = []
    #     for i, (index, row) in enumerate(sampled_df.iterrows()):
    #         hover_text = f"Cluster: {cluster_labels[i]}"
    #         for col in numerical_cols:
    #             hover_text += f"<br>{col}: {row[col]:.2f}"
    #         hover_texts.append(hover_text)
    #
    #     # Scatter plot for the first two principal components
    #     scatter = go.Scattergl(x=pca_result[:, 0], y=pca_result[:, 1],
    #                            mode='markers',
    #                            marker=dict(color=cluster_labels, colorscale='Viridis'),
    #                            text=hover_texts,  # Updated hover text
    #                            hoverinfo='text')
    #
    #     layout = go.Layout(title=f'{self.n_cluster_for_kmeans} Clusters of similar values (Kmeans)',
    #                        xaxis=dict(title='Direction of highest variance'),
    #                        yaxis=dict(title='Direction of second highest variance'), hovermode='closest')
    #
    #     fig = go.Figure(data=[scatter], layout=layout)
    #     self.phase_5_dict['kmeans_plot'] = fig_to_json(fig)
    #
    # def use_datashader_for_kmeans(self, scaled_data):
    #     # Assuming scaled_data is your pre-processed data
    #     # Perform PCA with a dynamic number of components based on the data
    #     n_samples, n_features = scaled_data.shape
    #     n_components = min(10, n_samples, n_features)
    #
    #     pca = PCA(n_components=n_components)
    #     pca_result = pca.fit_transform(scaled_data)
    #
    #     # Create a DataFrame from PCA result for Datashader
    #     df_pca = pd.DataFrame(pca_result, columns=[f'dim_{i}' for i in range(pca_result.shape[1])])
    #     # Create a Canvas object
    #     cvs = ds.Canvas(plot_width=800, plot_height=800)
    #     agg = cvs.points(df_pca, 'dim_0', 'dim_1', ds.count())
    #
    #     # Generate the image
    #     img = tf.shade(agg, cmap=fire, how='eq_hist')
    #
    #     # Convert the image to JSON
    #     self.phase_5_dict['kmeans_datashader'] = json.dumps(self.xr_image_to_json(img))

    # def xr_image_to_json(self, img, title="Title"):
    #     """Converts a Datashader image to a JSON format that can be used in Plotly."""
    #     # Convert to numpy array
    #     img_data = np.array(img.to_pil(), dtype=np.uint8)
    #
    #     # Create a Plotly figure
    #     fig = {
    #         "data": [{
    #             "type": "heatmap",
    #             "z": img_data.tolist(),
    #             "colorscale": "Viridis"
    #         }],
    #         "layout": {
    #             "title": title,
    #             "xaxis": {"title": "X-axis"},
    #             "yaxis": {"title": "Y-axis"}
    #         }
    #     }
    #     return fig

    def fifth_phase(self):
        if self.skip_table_stats == 'Y':
            return {}
        self.phase_5_dict = {}

        # Downsize numerical columns and convert to float32 or int32
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = self.df[col].astype('float32')
        for col in self.df.select_dtypes(include=['int64']).columns:
            self.df[col] = self.df[col].astype('int32')

        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=['int32', 'float32']).columns.tolist()
        if self.sample_size:
            # Sample data if it's too large
            self.sample_size = min(self.sample_size, len(self.df))  # Adjust this number based on your needs
            sampled_df = self.df.sample(n=self.sample_size, random_state=42) if len(
                self.df) > self.sample_size else self.df
        else:
            self.sample_size = len(self.df)
            sampled_df = self.df

        try:
            imputer = SimpleImputer(strategy='mean')
            imputed_data = imputer.fit_transform(sampled_df[numerical_cols])
            scaled_data = StandardScaler().fit_transform(imputed_data)
            # Correlation Analysis
            corr = sampled_df[numerical_cols].corr()
            heatmap_fig = go.Figure(
                data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
            heatmap_fig.update_layout(title='Correlation Heatmap')
            self.phase_5_dict['correlation_heatmap'] = fig_to_json(heatmap_fig)

        except Exception as e:
            print("Failed Correlation", e)

        try:
            # Preprocessing steps like imputation and scaling
            imputer = SimpleImputer(strategy='mean')
            imputed_data = imputer.fit_transform(self.df[numerical_cols])
            scaled_data = StandardScaler().fit_transform(imputed_data)

            # Check if enough samples are available for PCA
            n_samples, n_features = scaled_data.shape
            if n_samples < 2:
                raise ValueError("Not enough samples for PCA.")

            # print("Scaled data shape:", scaled_data.shape)
            n_components = min(10, n_samples, n_features)
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)

            # Aggregate loadings across all components for each feature
            cumulative_loadings = np.abs(pca.components_).sum(axis=0)

            # Adjusting the index for feature_importance to match scaled_data's features
            scaled_numerical_cols = self.df[numerical_cols].iloc[:, :n_features].columns
            feature_importance = pd.Series(cumulative_loadings, index=scaled_numerical_cols)
            # PCA Feature Importance
            # Ensure to round data to 2 decimal places before plotting
            sorted_importance = feature_importance.sort_values(ascending=False).round(2)

            # Create a bar plot for the feature contributions to PCA
            pca_bar_fig = go.Figure(data=go.Bar(x=sorted_importance.index, y=sorted_importance.values))
            pca_bar_fig.update_layout(title='Feature Contributions to PCA Variance',
                                      xaxis_title='Column Names', yaxis_title='Cumulative PCA Loadings')

            # Convert the Plotly figure to JSON
            self.phase_5_dict['pca_feature_importance'] = fig_to_json(pca_bar_fig)

        except Exception as e:
            print("Failed PCA", e)

        # try:
        #     # KMeans Clustering with PCA
        #     if len(self.df) >= self.plot_threshold_rows:  # Use Datashader for large datasets
        #         self.use_datashader_for_kmeans(scaled_data)
        #     else:
        #         self.use_plotly_for_kmeans(scaled_data, sampled_df, numerical_cols)
        # except Exception as e:
        #     print("Failed KMeans", e)

    # Define a function to generate the HTML report
    def generate_html_report(self):
        """
        Generates an HTML report from the data collected in the data profiling phases.
        """
        self.jinja_data = {
            'phase1_data': self.phase_1_dict,
            'custom_data_types': self.custom_data_types,
            'phase2_data': self.phase_2_dict,
            'categorical_info': self.categorical_info,
            'phase3_data': convert_to_serializable(self.phase_3_list),
            'phase4_data': {col: convert_to_serializable(data) for col, data in
                            self.phase_4_dict.items()} if self.phase_4_dict else {},
            'phase5_data': self.phase_5_dict,
            'random_rows': self.random_rows_df,
            'number_of_samples': self.sample_size
        }
        template = self.env.get_template(self.template_name)
        html_content = template.render(self.jinja_data, get_confidence_color=get_confidence_color)

        # Minify HTML (including inline JS and CSS)
        minified_html = minify_html(html_content)

        # Generate the output file path
        output_filepath = os.path.join('./', self.output_file_name)

        # Compress and save the minified HTML content as a .gz file
        compress_html(minified_html, output_filepath)

        print(f"Compressed HTML report generated: {output_filepath}.gz\n")


if __name__ == '__main__':
    # Specify the folder path
    folder_path = 'data'

    # List all the files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through each file
    for file_name in file_list:
        # Check if the file is a CSV file and matches the naming convention
        if file_name.endswith('.csv') and file_name.startswith('input_'):
            # Extract the schema and table name from the file name
            # input_mdjpndl_JP_STG_BtB_AccountMaster.csv
            schema_name = file_name.split('_')[1]
            table_name = ("_".join(file_name.split('_')[2:])).split('.')[0]
            # print(schema_name,table_name)
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Create a DataProfile object with schema and table name
            data_profiler = DataProfile(file_path=file_path, env_name='prod', schema_name=schema_name,
                                        table_name=table_name, output_folder='.', skip_col_stats='N',
                                        skip_table_stats='N')
            data_profiler.first_phase()
            data_profiler.second_phase()
            data_profiler.third_phase()
            data_profiler.fourth_phase()
            data_profiler.fifth_phase()
            data_profiler.generate_html_report()
            # Save format cache for future use
            with open(format_cache_file, 'wb') as f:
                pickle.dump(format_cache, f)
            print("\nDone", file_path, "\n")
