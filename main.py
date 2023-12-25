import os
import re
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import base64
from io import BytesIO
import dateparser
import fasttext
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import entropy, t, chi2, probplot, shapiro, norm, ttest_1samp, chisquare
from statsmodels.stats.proportion import proportion_confint
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import dateutil.parser as dparser
from jinja2 import Environment, FileSystemLoader
# from weasyprint import HTML
import os

# Load the fasttext model (make sure to provide the correct path to the model file)
ft_model = fasttext.load_model('data/lid.176.bin')

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


def detect_language_with_confidence(text):
    """
    Detect the language of the given text using FastText.
    :param text: Text to detect language for.
    :return: List of tuples with language code and confidence.
    """
    # Clean the text data
    cleaned_text = re.sub(r'\W+', ' ', text)

    try:
        # Predict languages with their probabilities
        predictions = ft_model.predict(cleaned_text, k=1)
        languages = predictions[0]
        probabilities = predictions[1]
        return [(lang.replace('__label__', ''), round(prob, 2)) for lang, prob in zip(languages, probabilities)]
    except Exception as e:
        return [('unknown', 0.0)]


def calculate_entropy(column: pd.Series) -> float:
    value_counts = column.value_counts()
    probabilities = value_counts / len(column)
    return entropy(probabilities)


def categorical_confidence(col_data: pd.Series, total_rows: int) -> float:
    unique_values = col_data.nunique()
    unique_ratio = unique_values / total_rows
    if unique_values <= 1:  # Single or no value
        return 0.0
    elif col_data.dtype in ['object', 'category'] or unique_ratio < 0.1:
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
    date_time = datetime.now().strftime("%m-%d-%Y | %I-%M %p")
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
    n_rows = min(10, df.shape[0])
    random_rows_df = df.sample(n=n_rows)
    memory_usage = df.memory_usage(deep=True).sum() / 1024
    info_dict = {
        'Environment': env,
        'Schema Name': schema_name,
        'Table Name': table_name,
        'Date/Time': date_time,
        'Shape': shape,
        'Column Names': column_names,
        'Custom Data Types': custom_data_types,
        'Has Duplicates': 'Yes' if has_duplicates else 'No',
        'Memory Usage': f'{memory_usage:.1f} KB',
        'Data Volume': f'{data_volume:.1f} KB',
        'Total RAM': f'{total_ram:.1f} GB',
        'Available RAM': f'{available_ram:.1f} GB',
        'Total Row Count': df.shape[0],
        'Total Column Count': df.shape[1]
    }
    return info_dict, random_rows_df, df


# # Get DataFrame info and random rows
df_info, random_rows_df, df = first_phase('data/input_data.csv', 'prod', 'schema', 'table')

# Iterate through the info dictionary and print the values
for key, value in df_info.items():
    print(f"{key}: {value}")

# Print the random rows DataFrame
print("\nRandom Rows:")
print(random_rows_df)


def second_phase(df: pd.DataFrame, custom_data_types: dict):
    int_count = 0
    float_count = 0
    date_count = 0
    timestamp_count = 0
    string_count = 0
    double_count = 0
    categorical_info = {}
    max_string_length = 0
    max_decimal_places = 0
    total_zero_percent_count = 0
    total_hundred_percent_count = 0
    total_rows = df.shape[0]
    for col in df.columns:
        col_data = df[col].dropna()
        data_type = custom_data_types[col]
        if data_type == 'integer':
            int_count += 1
        elif data_type == 'float':
            float_count += 1
            decimals = col_data.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0)
            max_decimal_places_col = decimals.max()
            max_decimal_places = max(max_decimal_places, max_decimal_places_col)
            if max_decimal_places_col > 6:
                double_count += 1
        elif data_type in ['string', 'category']:
            string_count += 1
            max_length_in_col = col_data.astype(str).map(len).max()
            max_string_length = max(max_string_length, max_length_in_col)
            if categorical_confidence(col_data, total_rows) > 50:
                categorical_info[col] = categorical_confidence(col_data, total_rows)
        elif data_type == 'date':
            date_count += 1
        elif data_type == 'timestamp':
            timestamp_count += 1
        if col_data.empty:
            total_zero_percent_count += 1
        elif col_data.count() == total_rows:
            total_hundred_percent_count += 1
    total_null_count = df.isnull().sum().sum()
    total_not_null_count = df.notnull().sum().sum()
    info_dict = {
        'Total Data Type Count': len(custom_data_types),
        'Int Column Count': int_count,
        'Float Column Count': float_count,
        'String Column Count': string_count,
        'Date Column Count': date_count,
        'Timestamp Column Count': timestamp_count,
        'Double Column Count': double_count,
        'Categorical Columns and Confidence Levels': categorical_info,
        'Maximum String Length': max_string_length,
        'Maximum Decimal Places': max_decimal_places,
        'Total Null Record Count': total_null_count,
        'Total Not Null Record Count': total_not_null_count,
        'Total number of 0% Record Count': total_zero_percent_count,
        'Total number of 100% Record Count': total_hundred_percent_count
    }
    return info_dict


# Get second phase info
second_phase_info = second_phase(df, df_info['Custom Data Types'])

# Iterate through the info dictionary and print the values
for key, value in second_phase_info.items():
    print(f"{key}: {value}")

print()
print()


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
            "Uniqueness Index": unique_values / total_count,
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
            non_english_chars = any(re.search(r'[^\x00-\x7F]+', str(x)) for x in col_data)
            column_info["Contains Non-English Characters"] = non_english_chars

            if non_english_chars:
                unique_texts = col_data.dropna().unique().tolist()
                languages_with_confidence = [detect_language_with_confidence(text) for text in unique_texts]
                column_info["Languages Detected with Confidence"] = languages_with_confidence
            else:
                column_info["Languages Detected with Confidence"] = [("EN", 1.0)]

        if "date" in data_type or "timestamp" in data_type:
            col_data = pd.to_datetime(col_data, errors='coerce')
            column_info["Min Date/Time Value"] = col_data.min()
            column_info["Max Date/Time Value"] = col_data.max()

        column_details.append(column_info)

    return column_details


# Get third phase info
third_phase_info = third_phase(df, df_info['Custom Data Types'])

# Iterate over the list of dictionaries
for column_info in third_phase_info:
    # Print the column name
    print(f"Column Name: {column_info['Column Name']}")

    # Iterate over the keys and values in the dictionary
    for key, value in column_info.items():
        # Skip the column name as we have already printed it
        if key != 'Column Name':
            print(f"{key}: {value}")

    # Print a separator for readability
    print("-" * 50)


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
    return image_base64


# Function to calculate and plot word cloud
def plot_wordcloud(text, ax):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')


# Function to calculate and plot TF-IDF
def calculate_tfidf(col_data, ax):
    # Clean the data
    col_data = col_data.apply(lambda x: re.sub(r'\W+', ' ', x.lower()) if isinstance(x, str) else x)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(col_data)
    tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    top_features = tfidf_scores.mean().sort_values(ascending=False).head(10)
    top_features.plot(kind='bar', ax=ax)
    ax.set_title('Top 10 TF-IDF Features')
    return top_features.to_dict()


# Function to calculate outlier percentage
def calculate_outlier_percentage(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_percentage = len(outliers) / len(data) * 100
    return outlier_percentage


# Fourth phase function
def fourth_phase(df, custom_data_types):
    column_analysis = {}

    for col in df.columns:
        data = df[col].dropna()
        dtype = custom_data_types[col]

        column_info = {}
        fig, ax = plt.subplots()

        if dtype == 'integer' or dtype == 'float':
            # Histogram
            plot_histogram(data, ax, f'Distribution of {col}')
            column_info['histogram'] = get_image_base64(fig)

            # QQ Plot
            fig, ax = plt.subplots()
            probplot(data, dist="norm", plot=ax)
            ax.set_title(f'QQ Plot of {col}')
            column_info['qq_plot'] = get_image_base64(fig)

            # Box Plot
            fig, ax = plt.subplots()
            plot_boxplot(data, ax, f'Boxplot of {col}')
            column_info['box_plot'] = get_image_base64(fig)

            # Violin Plot
            fig, ax = plt.subplots()
            plot_violinplot(data, ax, f'Violin plot of {col}')
            column_info['violin_plot'] = get_image_base64(fig)

            # Outlier Percentage
            outlier_percentage = calculate_outlier_percentage(data)
            column_info['outlier_percentage'] = outlier_percentage

        elif dtype == 'string':
            # Word Cloud
            combined_text = ' '.join(data.astype(str))
            fig, ax = plt.subplots()
            plot_wordcloud(combined_text, ax)
            column_info['wordcloud'] = get_image_base64(fig)

            # TF-IDF Ranking
            tfidf_scores = calculate_tfidf(data, ax)
            column_info['tfidf'] = tfidf_scores

            # Histogram of text length
            fig, ax = plt.subplots()
            text_lengths = data.apply(len)
            plot_histogram(text_lengths, ax, f'Text Length Distribution for {col}')
            column_info['text_length_histogram'] = get_image_base64(fig)

        elif dtype == 'date' or dtype == 'timestamp':
            # Histogram with dynamic binning
            fig, ax = plt.subplots()
            data = pd.to_datetime(data, errors='coerce')
            data.dropna().hist(ax=ax, bins=20)
            ax.set_title(f'Distribution of {col}')
            column_info['date_hist'] = get_image_base64(fig)

        # Save the column info
        column_analysis[col] = column_info

    return column_analysis


# Set up Jinja2 environment
# Replace 'your_templates_directory' with the actual path to your templates
env = Environment(loader=FileSystemLoader('./'))


# Define a function to generate the HTML report
def generate_html_report(data_phases, template_name='jinja_tempate.html',
                         output_filename='data_profiling_report.html'):
    """
    Generates an HTML report from the data collected in the data profiling phases.

    :param data_phases: A dictionary containing the data for each phase.
    :param template_name: The filename of the Jinja2 template.
    :param output_filename: The filename for the output HTML.
    """
    # Load the Jinja2 template
    template = env.get_template(template_name)

    # Use the template to render HTML content
    html_content = template.render(data_phases)

    # Define the output path for the HTML file
    output_filepath = os.path.join('path_to_output_directory', output_filename)

    # Save the rendered HTML to a file
    with open(output_filepath, 'w') as file:
        file.write(html_content)

    # Return the path to the generated HTML file
    return output_filepath

# Example usage:
# Assuming 'data_phases' is a dictionary containing all the necessary data and visualizations from the four phases
data_phases = {
    'phase1': first_phase('data/input_data.csv', 'prod', 'schema', 'table'),
    'phase2': second_phase(df, df_info['Custom Data Types']),
    'phase3': third_phase(df, df_info['Custom Data Types']),
    'phase4': fourth_phase(df, df_info['Custom Data Types'])
}
html_report_path = generate_html_report(data_phases)
print(f"HTML report generated: {html_report_path}")