# Standard library imports
import base64
import contextlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import Union, Any, Dict
import functools
import multiprocessing
from fitter import Fitter
import numpy as np
from functools import partial
import pandas
import unicodedata
import multiprocessing
import numpy as np
from scipy import stats
from scipy.stats import norm, lognorm, expon, weibull_min, gamma
import warnings
# Third-party libraries
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.decomposition import PCA
import seaborn as sns
import pycountry
import dateparser
from dateutil import parser as dutil_parser
import fasttext
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import Histogram, Box, Scatter, Figure, Pie, Bar
import psutil
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import entropy, skew
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from fitter import Fitter, get_common_distributions
import textstat
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import io
# Load the fasttext model (make sure to provide the correct path to the model file)
with contextlib.redirect_stderr(io.StringIO()):
    ft_model = fasttext.load_model('data/lid.176.ftz')
pd.set_option('display.max_columns', None)
# Compile the regex pattern outside of the function
non_word_pattern = re.compile(r'\W+')


# Ensure you have the appropriate datasets downloaded
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

def custom_data_type(col):
    if col.isnull().all():
        return 'empty'
    if pd.api.types.is_numeric_dtype(col):
        if (col.dropna() % 1 == 0).all():
            return 'integer'
        else:
            return 'float'
    if pd.api.types.is_string_dtype(col):
        non_null_col = col.dropna()
        if non_null_col.empty:
            return 'empty'

        # Attempt to parse the entire column
        parsed_col = pd.to_datetime(non_null_col, errors='coerce')

        # Check if any value failed to parse
        if parsed_col.isna().any():
            return 'string'

        # All values parsed successfully; determine if date or timestamp
        if (parsed_col.dt.hour == 0).all() and (parsed_col.dt.minute == 0).all() and (parsed_col.dt.second == 0).all():
            return 'date'
        else:
            return 'timestamp'

    return 'string'


def contains_non_english_characters(col_data: pd.Series) -> bool:
    # Convert series to string, then use a vectorized operation to check for non-English characters
    non_english_chars = col_data.dropna().astype(str).apply(lambda x: not x.isascii())
    return non_english_chars.any()


@functools.lru_cache(maxsize=128)
def detect_language_with_confidence(text: str) -> list:
    # Check for non-Latin characters using unicodedata
    if any(unicodedata.category(char).startswith('L') and not unicodedata.name(char).startswith('LATIN') for char in
           text):
        # Clean the text data
        cleaned_text = non_word_pattern.sub(' ', text)
        try:
            # Predict languages with their probabilities
            predictions = ft_model.predict(cleaned_text, k=1)
            languages = predictions[0]
            probabilities = predictions[1]
            full_language_names = []
            for lang_code, prob in zip(languages, probabilities):
                lang_code = lang_code.replace('__label__', '')
                # Get the full language name from pycountry
                try:
                    language = pycountry.languages.get(alpha_2=lang_code)
                    full_language_names.append((language.name, round(prob * 100, 2)))
                except AttributeError:
                    # If the language code is not found in pycountry, return the code
                    full_language_names.append((lang_code, round(prob * 100, 2)))
            return full_language_names
        except Exception as e:
            return [('unknown', 0)]
    else:
        # If text contains only Latin characters, assume it's English or another Latin-based language
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
    return len(word_tokenize(text)) >= min_word_count


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
    """Convert numpy types and nested collections to native Python types for JSON serialization."""
    if isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, (dict, list, tuple)):
        if isinstance(val, dict):
            return {k: convert_to_serializable(v) for k, v in val.items()}
        else:  # For lists and tuples
            return type(val)(convert_to_serializable(v) for v in val)
    else:  # No need for conversion if it's already a native Python type
        return val


class DataProfile:
    def __init__(self, file_path, env_name='dummy', schema_name='dummy_schema', table_name='dummy_table',
                 output_folder='.'):

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
        self.pos_tag_descriptions = {
            'CC': 'Coordinating conjunction',
            'CD': 'Cardinal number',
            'DT': 'Determiner',
            'EX': 'Existential there',
            'FW': 'Foreign word',
            'IN': 'Preposition or subordinating conjunction',
            'JJ': 'Adjective',
            'JJR': 'Adjective, comparative',
            'JJS': 'Adjective, superlative',
            'LS': 'List item marker',
            'MD': 'Modal',
            'NN': 'Noun, singular or mass',
            'NNS': 'Noun, plural',
            'NNP': 'Proper noun, singular',
            'NNPS': 'Proper noun, plural',
            'PDT': 'Predeterminer',
            'POS': 'Possessive ending',
            'PRP': 'Personal pronoun',
            'PRP$': 'Possessive pronoun',
            'RB': 'Adverb',
            'RBR': 'Adverb, comparative',
            'RBS': 'Adverb, superlative',
            'RP': 'Particle',
            'SYM': 'Symbol',
            'TO': 'to',
            'UH': 'Interjection',
            'VB': 'Verb, base form',
            'VBD': 'Verb, past tense',
            'VBG': 'Verb, gerund or present participle',
            'VBN': 'Verb, past participle',
            'VBP': 'Verb, non-3rd person singular present',
            'VBZ': 'Verb, 3rd person singular present',
            'WDT': 'Wh-determiner',
            'WP': 'Wh-pronoun',
            'WP$': 'Possessive wh-pronoun',
            'WRB': 'Wh-adverb'}
        # Set up Jinja2 environment
        self.env = Environment(loader=FileSystemLoader('.'))
        self.template_name = 'jinja_template.html'
        self.output_folder = output_folder
        self.output_file_name = os.path.join(self.output_folder,
                                             'data_profiling_report_' + self.schema_name + "_" + self.table_name + '_' + datetime.now().strftime(
                                                 "%d%m%Y_%H_%M_%S") + '.html')

    # Custom function to determine the data type of a column

    def first_phase(self):
        date_time: str = datetime.now().strftime("%m/%d/%Y %I:%M %p")
        data_volume: float = os.path.getsize(self.file_path) / 1024
        total_ram: Union[float, Any] = psutil.virtual_memory().total / (1024 ** 3)
        available_ram: Union[float, Any] = psutil.virtual_memory().available / (1024 ** 3)
        chunksize = 10 ** 6 if data_volume > available_ram else None
        if chunksize:
            df_chunks = pd.read_csv(self.file_path, chunksize=chunksize)
            self.df = pd.concat(df_chunks, ignore_index=True)
        else:
            self.df = pd.read_csv(self.file_path)
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
        self.custom_data_types: dict[Any, str] = {col: custom_data_type(self.df[col]) for col in self.df.columns}
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
            elif data_type in ['string', 'category']:
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

    def fourth_phase(self):
        self.phase_4_dict = {}

        for col in self.df.columns:
            data: pandas.DataFrame = self.df[col].dropna()
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
                'max': 'Maximum Value'}
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

                # # Find best fit distribution
                # best_fit_results = best_fit_distribution(data)  # Adjust the number of bins as needed
                # if best_fit_results:
                #     best_fit_name = best_fit_results[0]  # Get the name of the best fit distribution
                # else:
                #     best_fit_name = None  # Or handle the case where no best fit was found
                #
                # column_info['description']['distribution'] = best_fit_name

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

            elif dtype == 'date' or dtype == 'timestamp':

                date_data = pd.to_datetime(data)
                try:
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

                try:
                    # Text analysis
                    full_text = ' '.join(data).lower()
                    # Remove all symbols
                    cleaned_text = re.sub(r'^\w\s', '', full_text)
                    # Replace multiple spaces, tabs, and new lines with a single space
                    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                    full_text = cleaned_text.strip()
                    column_info['description']['flesch reading score'] = round(np.mean(
                        [textstat.flesch_reading_ease(text) if is_valid_text(text) else 0 for text in data]), 2)
                    column_info['description']['character count'] = textstat.char_count(full_text, ignore_spaces=True)
                    column_info['description']['polysyllable count'] = textstat.polysyllabcount(full_text)
                    column_info['description']['monosyllable count'] = textstat.monosyllabcount(full_text)

                except Exception as e:
                    print(col, e)

                try:
                    # Assuming `data` is a list of strings (documents) from your dataframe column
                    full_text = ' '.join(data)
                    # Tokenize the text into words
                    tokens = word_tokenize(full_text)
                    # Get the list of POS tags
                    tags = pos_tag(tokens)
                    # Count the frequency of each part of speech
                    pos_counts = Counter(tag for word, tag in tags)
                    # Prepare x and y data for the bar chart
                    x_data = list(pos_counts.keys())
                    y_data = list(pos_counts.values())
                    hover_text = [f"{self.pos_tag_descriptions.get(tag, 'Symbols')}: {count:,}" for tag, count in
                                  pos_counts.items()]

                    # Create the bar chart
                    fig = go.Figure(data=[go.Bar(x=x_data, y=y_data, text=hover_text, hoverinfo='text')])

                    # Update layout to ensure x-axis labels (POS tags) are shown
                    fig.update_layout(
                        title=f"Parts of Speech Tag Counts for {col}",
                        xaxis=dict(
                            title="POS Tags",
                            tickmode='array',
                            tickvals=x_data,
                            ticktext=x_data,
                            tickangle=45,  # Rotate the labels to prevent overlap
                            showticklabels=True  # Ensure that labels are shown
                        ),
                        yaxis=dict(
                            title="Count"
                        ),
                        margin=dict(  # Add margins to the layout to ensure labels are not cut off
                            l=50,
                            r=50,
                            b=100,  # Increase bottom margin to accommodate rotated labels
                            t=50,
                            pad=4
                        ),
                        autosize=False,  # Disable autosize to maintain the set figure size
                        height=600,  # Set figure height to ensure labels fit
                        width=800  # Set figure width as necessary
                    )

                    # Serialize the POS counts bar chart to JSON
                    column_info['pos_counts_bar_chart'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                except Exception as e:
                    print(col, e)
            self.phase_4_dict[col] = column_info

    def fifth_phase(self, skip='N'):
        if skip == 'Y':
            return {}
        self.phase_5_dict = {}

        # Segregate columns by type
        numerical_cols = []
        categorical_cols = []
        date_time_cols = []

        for col in self.df.columns:
            col_type = custom_data_type(self.df[col])
            if col_type in ['integer', 'float']:
                numerical_cols.append(col)
            elif col_type in ['string']:
                if categorical_confidence(self.df[col], len(self.df)) > 0:
                    categorical_cols.append(col)
            elif col_type in ['date', 'timestamp']:
                date_time_cols.append(col)

        # Numerical Analysis
        if numerical_cols:
            # Handling NaN values for numerical columns
            imputer = SimpleImputer(strategy='mean')  # or median, or most_frequent
            imputed_data = imputer.fit_transform(self.df[numerical_cols])
            scaled_data = StandardScaler().fit_transform(imputed_data)

            # Correlation Analysis
            corr = self.df[numerical_cols].corr()
            heatmap_fig = go.Figure(
                data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
            heatmap_fig.update_layout(title='Correlation Heatmap')
            self.phase_5_dict['correlation_heatmap'] = fig_to_json(heatmap_fig)

            # Perform PCA
            pca = PCA(n_components=len(numerical_cols))  # Use all components to get the full loading matrix
            pca.fit(scaled_data)

            # Retrieve the absolute value of loadings of the first principal component
            loadings = np.abs(pca.components_[0])

            # Create a series with column names and their corresponding loadings
            loading_series = pd.Series(loadings, index=numerical_cols)

            # Sort the series to identify the most important features
            sorted_loadings = loading_series.sort_values(ascending=False)

            # Select the top N features to display in the bar chart, you can choose N based on your preference
            top_features = sorted_loadings.head(10)

            # Create a bar chart with Plotly
            pca_bar_fig = go.Figure(data=go.Bar(x=top_features.index, y=top_features.values))
            pca_bar_fig.update_layout(title='Columns ranked according to contribution to overall variance (PCA)',
                                      xaxis_title='Features',
                                      yaxis_title='PCA Loadings')

            # Convert to JSON using fig_to_json
            self.phase_5_dict['pca_feature_importance'] = fig_to_json(pca_bar_fig)

            # Perform PCA to reduce the data to 2 dimensions for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)

            # Running KMeans clustering
            kmeans = KMeans(n_clusters=3, n_init=10)
            cluster_labels = kmeans.fit_predict(pca_result)

            # Prepare hover text
            hover_text = []
            for i in range(pca_result.shape[0]):
                text = ''
                for j, column in enumerate(self.df[numerical_cols].columns):
                    text += f'{column}: {self.df[numerical_cols].iloc[i, j]:.2f}<br>'
                hover_text.append(text)

            # Creating the scatter plot for the first two principal components
            scatter = go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1],
                                 mode='markers',
                                 marker=dict(color=cluster_labels, colorscale='Viridis', showscale=False),
                                 text=hover_text,
                                 hoverinfo='text')

            # Setting the layout for the plot
            layout = go.Layout(title='Similar datapoints clustering using K-Means Clustering',
                               xaxis=dict(title='Direction of highest variance'),
                               yaxis=dict(title='Direction of next highest variance'),
                               hovermode='closest')

            # Creating the figure
            fig = go.Figure(data=[scatter], layout=layout)
            self.phase_5_dict['kmeans_plot'] = fig_to_json(fig)

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
            'phase3_data': self.phase_3_list,
            'phase4_data': {col: convert_to_serializable(data) for col, data in self.phase_4_dict.items()},
            'phase5_data': self.phase_5_dict,
            'random_rows': self.random_rows_df
        }
        template = self.env.get_template(self.template_name)
        html_content = template.render(self.jinja_data, get_confidence_color=get_confidence_color)
        output_filepath = os.path.join('./', self.output_file_name)

        # Save the rendered HTML to a file
        with open(output_filepath, 'w', encoding='utf-8') as file:
            file.write(html_content)

        print(f"HTML report generated: {output_filepath}\n")


if __name__ == '__main__':
    # Specify the folder path
    folder_path = 'full_data'

    # List all the files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through each file
    for file_name in file_list:
        # Check if the file is a CSV file and matches the naming convention
        if file_name.endswith('.csv') and file_name.startswith('input_'):
            # Extract the schema and table name from the file name
            #input_mdjpndl_JP_STG_BtB_ConsignmentManagementData.csv
            schema_name = file_name.split('_')[1]
            table_name = "_".join(file_name.split('_')[2:-1]).split('.')[0]

            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Run the provided code for each file with schema and table name
            if __name__ == '__main__':
                # Create a DataProfile object with schema and table name
                data_profiler = DataProfile(file_path=file_path, env_name='prod', schema_name=schema_name,
                                            table_name=table_name, output_folder='full_output')
                data_profiler.first_phase()
                data_profiler.second_phase()
                data_profiler.third_phase()
                data_profiler.fourth_phase()
                data_profiler.fifth_phase(skip='Y')
                data_profiler.generate_html_report()
                print("\nDone",file_path,"\n")
