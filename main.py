import os
import time
import warnings

from Data_Profiler_TCS import Profiler

warnings.filterwarnings("ignore")

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def profile_csv(file_path, schema_name='schema_name', table_name='table_name',
                output_folder='output'):
    """Profile a single CSV file and generate an HTML report."""
    start_time = time.time()

    output_path = os.path.join(SCRIPT_DIR, output_folder)
    os.makedirs(output_path, exist_ok=True)

    data_profiler = Profiler.DataProfile(
        file_path=file_path,
        env_name='prod',
        schema_name=schema_name,
        table_name=table_name,
        output_folder=output_path,
        skip_col_stats='N',
        skip_table_stats='N',
    )
    data_profiler.first_phase()
    data_profiler.second_phase()
    data_profiler.third_phase()
    data_profiler.fourth_phase()
    data_profiler.fifth_phase()

    report_path = data_profiler.generate_html_report()
    report_size_bytes = os.path.getsize(report_path)
    report_size_kb = report_size_bytes / 1024
    report_size_mb = report_size_kb / 1024

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    row_count = len(data_profiler.df)
    column_count = len(data_profiler.df.columns)
    print(
        f"\nDone {file_path}"
        f"\nRows: {row_count}"
        f"\nColumns: {column_count}"
        f"\nTime Taken: {int(hours)}h {int(minutes)}m {int(seconds)}s"
    )
    print(f"Report Size: {report_size_mb:.2f} MB / {report_size_kb:.2f} KB\n")


def main():
    folder_path = os.path.join(SCRIPT_DIR, 'data')

    if not os.path.isdir(folder_path):
        print(f"Data folder not found: {folder_path}")
        return

    file_list = os.listdir(folder_path)

    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            profile_csv(file_path)


if __name__ == "__main__":
    main()
