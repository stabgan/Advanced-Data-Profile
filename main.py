import base64
import json
# from weasyprint import HTML
import os
import re
from datetime import datetime
from io import BytesIO
import pycountry
import dateparser
import dateutil.parser as dparser
import fasttext
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
import psutil
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import entropy, skew
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import unicodedata
import json
from plotly import offline as py_offline
from plotly.graph_objs import Histogram, Box, Scatter, Figure
# Load the fasttext model (make sure to provide the correct path to the model file)
ft_model = fasttext.load_model('data/lid.176.bin')
pd.set_option('display.max_columns', None)

schema_name = "schema"
table_name = "table"


# Custom function to determine the data type of a column
def custom_data_type(col):
    if col.isnull().all():
        return 'empty'
    if pd.api.types.is_numeric_dtype(col):
        if (col.dropna() % 1 == 0).all():
            return 'integer'
        else:
            return 'float'
    if pd.api.types.is_string_dtype(col):
        try:
            first_value = col.dropna().iloc[0]
            parsed_date = dparser.parse(first_value, fuzzy=False)
            if parsed_date.hour == 0 and parsed_date.minute == 0 and parsed_date.second == 0:
                return 'date'
            else:
                return 'timestamp'
        except (ValueError, TypeError):
            return 'string'
    return 'string'


def contains_non_english_characters(col_data: pd.Series) -> bool:
    for item in col_data.dropna().astype(str):
        for char in item:
            if unicodedata.category(char).startswith('L') and (unicodedata.name(char).startswith('LATIN') is False):
                return True
    return False


def detect_language_with_confidence(text: str) -> list:
    # Check for non-Latin characters using unicodedata
    if any(unicodedata.category(char).startswith('L') and not unicodedata.name(char).startswith('LATIN') for char in
           text):
        # Clean the text data
        cleaned_text = re.sub(r'\W+', ' ', text)
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
                    full_language_names.append((language.name, round(prob, 2)))
                except AttributeError:
                    # If the language code is not found in pycountry, return the code
                    full_language_names.append((lang_code, round(prob, 2)))
            return full_language_names
        except Exception as e:
            return [('unknown', 0.0)]
    else:
        # If text contains only Latin characters, assume it's English or another Latin-based language
        return [('English', 1.0)]


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
        col_data = col_data.apply(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))
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


def first_phase(file_path: str, env: str, schema_name: str, table_name: str):
    date_time = datetime.now().strftime("%m/%d/%Y %I:%M %p")
    data_volume = os.path.getsize(file_path) / 1024
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    available_ram = psutil.virtual_memory().available / (1024 ** 3)
    chunksize = 10 ** 6 if data_volume > available_ram else None
    if chunksize:
        df_chunks = pd.read_csv(file_path, chunksize=chunksize)
        df = pd.concat(df_chunks, ignore_index=True)
    else:
        df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    custom_data_types = {col: custom_data_type(df[col]) for col in df.columns}
    shape = df.shape
    column_names = df.columns.tolist()
    has_duplicates = df.duplicated().any()
    n_rows = min(100, df.shape[0])
    random_rows_df = df.sample(n=n_rows)

    # Transpose the DataFrame
    # random_rows_transposed = random_rows_df.T
    memory_usage = df.memory_usage(deep=True).sum() / 1024
    info_dict = {
        'Environment': env,
        'Schema Name': schema_name,
        'Table Name': table_name,
        'Date/Time': date_time,
        'Has Duplicates': 'Yes' if has_duplicates else 'No',
        'Memory Usage in server': f'{memory_usage:.1f} KB',
        'Data volume': f'{data_volume:.1f} KB',
        'Total RAM in server': f'{total_ram:.1f} GB',
        'Available RAM in server': f'{available_ram:.1f} GB',
        'Total Row Count': df.shape[0],
        'Total Column Count': df.shape[1]
    }
    return info_dict, random_rows_df, df, custom_data_types


# # # Get DataFrame info and random rows
# df_info, random_rows_df, df = first_phase('data/input_data.csv', 'prod', 'schema', 'table')
#
# # Iterate through the info dictionary and print the values
# for key, value in df_info.items():
#     print(f"{key}: {value}")
#
# # Print the random rows DataFrame
# print("\nRandom Rows:")
# print(random_rows_df)


def second_phase(df, custom_data_types):
    # Initialize counters and lists to hold column names for each data type
    int_columns = []
    float_columns = []
    string_columns = []
    date_columns = []
    timestamp_columns = []
    double_columns = []
    categorical_info = {}
    max_string_length = 0
    max_decimal_places = 0
    total_zero_percent_count = 0
    total_hundred_percent_count = 0
    total_rows = df.shape[0]

    for col in df.columns:
        col_data = df[col].dropna()
        data_type = custom_data_types[col]
        categorical_info[col] = categorical_confidence(col_data, total_rows)
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

    total_null_count = df.isnull().sum().sum()
    total_not_null_count = df.notnull().sum().sum()
    # print(custom_data_types)

    info_dict = {
        'Total Data Type Count': len(set(custom_data_types.values())),
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

    return info_dict, categorical_info


# Get second phase info
# second_phase_info = second_phase(df, df_info['Custom Data Types'])
#
# # Iterate through the info dictionary and print the values
# for key, value in second_phase_info.items():
#     print(f"{key}: {value}")
#
# print()
# print()


def third_phase(df: pd.DataFrame, custom_data_types: dict):
    column_details = []

    for col in df.columns:
        col_data = df[col]
        data_type = custom_data_types[col]  # Using custom data type
        total_count = len(col_data)
        null_count = col_data.isnull().sum()
        non_null_count = total_count - null_count
        unique_values = col_data.nunique()
        top_10_values = col_data.value_counts().head(10).to_dict()

        column_info = {
            "Column Name": col,
            "Column Data Type": data_type,
            "NULL Record Count": null_count,
            "Percentage of NULL Values": null_count / total_count * 100,
            "NOT NULL Record Count": non_null_count,
            "Percentage of NOT NULL Values": non_null_count / total_count * 100,
            "Number of Distinct Values": unique_values,
            "Uniqueness Index (unique/total)": unique_values / total_count,
            "Top 10 Values": top_10_values
        }

        # Additional metrics for numeric and string data
        if data_type == 'integer' or data_type == 'float':
            column_info["Median"] = col_data.median()
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
                column_info["Languages Detected with Confidence"] = [("English", 1.0)]

        if "date" in data_type or "timestamp" in data_type:
            col_data = pd.to_datetime(col_data, errors='coerce')
            column_info["Min Date/Time Value"] = col_data.min()
            column_info["Max Date/Time Value"] = col_data.max()

        column_details.append(column_info)

    return column_details


# # Get third phase info
# third_phase_info = third_phase(df, df_info['Custom Data Types'])
#
# # Iterate over the list of dictionaries
# for column_info in third_phase_info:
#     # Print the column name
#     print(f"Column Name: {column_info['Column Name']}")
#
#     # Iterate over the keys and values in the dictionary
#     for key, value in column_info.items():
#         # Skip the column name as we have already printed it
#         if key != 'Column Name':
#             print(f"{key}: {value}")
#
#     # Print a separator for readability
#     print("-" * 50)


def plot_histogram(data, ax, title):
    sns.histplot(data, kde=False, ax=ax)
    ax.set_title(title)


def plot_boxplot(data, ax, title):
    sns.boxplot(x=data, ax=ax)
    ax.set_title(title)


def plot_violinplot(data, ax, title):
    sns.violinplot(x=data, ax=ax)
    ax.set_title(title)


def plot_qqplot(data, ax, title):
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(title)


# Function to encode plot as base64 for HTML embedding
def base64_encode_plot(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# Function to generate base64 encoded image for embedding in PDF
def get_image_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)  # Close the figure after encoding
    return image_base64


# Function to calculate and plot word cloud
# Function to calculate and plot word cloud
def plot_wordcloud(text):
    # Check if the text is empty
    if not text.strip():
        print("No data to generate word cloud for.")
        return None
    # If text is not empty, proceed with generating a word cloud image
    wordcloud = WordCloud(background_color='white').generate(text)

    # Save the image to a BytesIO object
    image_io = BytesIO()
    wordcloud.to_image().save(image_io, format='PNG')
    image_io.seek(0)

    # Encode the image to base64
    image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8')
    image_io.close()

    # Return the base64 encoded image
    return image_base64


# Function to calculate and plot TF-IDF
def calculate_tfidf(col_data):
    # Clean the data
    col_data = col_data.apply(lambda x: re.sub(r'\W+', ' ', x.lower().strip()) if isinstance(x, str) else x)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    if col_data.empty or col_data.str.strip().eq("").all():
        print("No data available for TF-IDF analysis.")
        return {}
    # Attempt to perform TF-IDF analysis, catch any errors that arise
    try:
        # Generate TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(col_data)
        tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        top_features = tfidf_scores.mean().sort_values(ascending=False).head(10)

        # Return the top features as a dictionary
        return top_features.to_dict()

    except ValueError as e:
        # Handle the case where TF-IDF could not be calculated, usually due to empty vocabulary
        print(f"Error in TF-IDF analysis: {e}")
        return {}

    # Function to calculate outlier percentage


def calculate_outlier_percentage(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_percentage = len(outliers) / len(data) * 100
    return outlier_percentage
def round_if_float(x):
    if isinstance(x, float):
        return round(x, 2)
    return x

# Fourth phase function
def fourth_phase(df, custom_data_types):
    column_analysis = {}

    for col in df.columns:
        data = df[col].dropna()
        dtype = custom_data_types[col]

        column_info = {'description': {k: round_if_float(v) for k, v in data.describe().to_dict().items()}}

        if dtype == 'integer' or dtype == 'float':
            # Basic statistical analysis
            column_info['skewness'] = round_if_float(data.skew())
            column_info['kurtosis'] = round_if_float(data.kurtosis())
            column_info['outlier_percentage'] = round_if_float(calculate_outlier_percentage(data))

            # Normalization and transformation (if needed)
            # Assuming a simple normalization here; can be expanded as needed
            normalized_data = (data - data.mean()) / data.std()
            column_info['normalized'] = normalized_data.tolist()

            # Outlier detection
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3].tolist()
            column_info['outliers'] = outliers

            # Preparing data for histograms, box plots, etc.
            column_info['histogram_data'] = data.tolist()
            column_info['boxplot_data'] = data.tolist()

            # Q-Q plot data
            qq_plot_data = stats.probplot(data, dist="norm")
            column_info['qq_plot'] = {'x': [val[0] for val in qq_plot_data[0]], 'y': [val[1] for val in qq_plot_data[0]]}

            # Violin plot data (using normalized data)
            column_info['violin_plot'] = normalized_data.tolist()

            # Cumulative frequency plot data
            sorted_data = np.sort(data)
            cum_freq = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            column_info['cumulative_freq'] = {'x': sorted_data.tolist(), 'y': cum_freq.tolist()}

            # Plotly figures
            # Histogram
            fig_hist = Figure(data=[Histogram(x=data)])
            column_info['histogram'] = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)

            # Box Plot
            fig_box = Figure(data=[Box(y=data)])
            column_info['boxplot'] = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)

            # Q-Q Plot
            # Q-Q plot data
            qq_plot_data = stats.probplot(data, dist="norm")

            # Check if qq_plot_data has the expected structure
            if len(qq_plot_data) == 2 and all(isinstance(i, np.ndarray) for i in qq_plot_data):
                qq_x, qq_y = zip(*[(val[0], val[1]) for val in qq_plot_data[0]])
                column_info['qq_plot'] = {'x': qq_x, 'y': qq_y}
            else:
                # Handle unexpected qq_plot_data format
                column_info['qq_plot'] = {'x': [], 'y': []}
                # You might want to log a warning or error here

            # Violin Plot
            fig_violin = Figure(data=[Box(y=normalized_data, boxpoints='all', jitter=0.5, pointpos=-1.8)])
            column_info['violin_plot'] = json.dumps(fig_violin, cls=plotly.utils.PlotlyJSONEncoder)

            # Cumulative Frequency Plot
            fig_cum_freq = Figure(data=[Scatter(x=sorted_data, y=cum_freq)])
            column_info['cumulative_freq'] = json.dumps(fig_cum_freq, cls=plotly.utils.PlotlyJSONEncoder)

        column_analysis[col] = column_info

    return column_analysis


    #     elif dtype == 'string':
    #         # Word Cloud
    #         if data.any():  # Check if there's any data
    #             combined_text = ' '.join(data.astype(str))
    #             try:
    #                 wordcloud_img = plot_wordcloud(combined_text)
    #             except ValueError as e:
    #                 print(f"Caught an error when generating word cloud: {e}")
    #                 wordcloud_img = None  # or set to a default image
    #             column_info['wordcloud'] = wordcloud_img
    #         else:
    #             print(f"No data to generate word cloud for column: {col}")
    #
    #         # tf idf scores
    #         tfidf_scores = calculate_tfidf(data)
    #         if tfidf_scores:
    #             # We convert the dictionary to a DataFrame for Plotly
    #             tfidf_df = pd.DataFrame(list(tfidf_scores.items()), columns=['feature', 'score'])
    #             fig = px.bar(tfidf_df, x='feature', y='score', title=f'TF-IDF Scores for {col}')
    #             column_info['tfidf_bar'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #         else:
    #             print(f"No TF-IDF scores to plot for {col}")
    #
    #         # Histogram of text length
    #         text_lengths = data.apply(len)
    #         fig = px.histogram(text_lengths, title=f'Text Length Distribution for {col}')
    #         column_info['text_length_histogram'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #
    #     elif dtype == 'date' or dtype == 'timestamp':
    #         # Check if there's any data after conversion to datetime
    #         data = pd.to_datetime(data, errors='coerce').dropna()
    #         if not data.empty:
    #             fig = px.histogram(data, title=f'Distribution of {col}')
    #             column_info['date_hist'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #         else:
    #             print(f"No valid date data for column: {col}")
    #
    #     column_analysis[col] = column_info
    #
    # print(f"Finished processing columns.")
    # return column_analysis


# Set up Jinja2 environment
# Replace 'your_templates_directory' with the actual path to your templates
env = Environment(loader=FileSystemLoader('.'))


# Define a function to generate the HTML report
def generate_html_report(data_phases, template_name='jinja_template.html',
                         output_filename='data_profiling_report.html'):
    """
    Generates an HTML report from the data collected in the data profiling phases.
    """
    template = env.get_template(template_name)
    html_content = template.render(data_phases, get_confidence_color=get_confidence_color)
    output_filepath = os.path.join('./', output_filename)

    # Save the rendered HTML to a file
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write(html_content)

    # Return the path to the generated HTML file
    return output_filepath


def get_confidence_color(confidence):
    # Assuming confidence is a percentage
    if confidence >= 85:
        return '#dc3545'  # Red for high confidence
    elif confidence >= 70:
        return '#007bff'  # Blue for medium confidence
    else:
        return '#28a745'  # Green for low confidence

def convert_to_serializable(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, np.generic):
        return val.item()  # Updated from np.asscalar(val) to val.item()
    elif isinstance(val, dict):
        return {k: convert_to_serializable(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [convert_to_serializable(v) for v in val]
    return val

df_info, random_rows, df, custom_data_types = first_phase('data/input_data.csv', 'prod', 'big_schema_name',
                                                          'ultra_huge_table0677_name')
summary, categorical_info = second_phase(df, custom_data_types)
phase3 = third_phase(df, custom_data_types)
# print(phase3)
phase4 = fourth_phase(df, custom_data_types)
# print(phase4)
phase4_serializable = {col: convert_to_serializable(data) for col, data in phase4.items()}
# print(phase4['summaryaccountcode'])
# random_rows = random_rows.fillna('<i>null/blank</i>')
# Example usage:
# print(random_rows)
# print(random_rows.to_html)
# Assuming 'data_phases' is a dictionary containing all the necessary data and visualizations from the four phases
data_phases = {
    'phase1_data': df_info,
    'custom_data_types': custom_data_types,
    'phase2_data': summary,
    'categorical_info': categorical_info,
    'phase3_data': phase3,
    'phase4_data': phase4_serializable,
    'random_rows': random_rows
}

# Generate the report
html_report_path = generate_html_report(data_phases, 'jinja_template.html')
print(f"HTML report generated: {html_report_path}")

# Make sure to close plots to avoid memory issues
plt.close('all')
