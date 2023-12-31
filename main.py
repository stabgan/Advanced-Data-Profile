import os
import time
import warnings

from Data_Profiler_TCS import Profiler

warnings.filterwarnings("ignore")

# Specify the folder path
folder_path = 'data'

# List all the files in the folder
file_list = os.listdir(folder_path)

# Iterate through each file
for file_name in file_list:
    # Check if the file is a CSV file and matches the naming convention
    # if file_name.endswith('.csv') and file_name.startswith('input_'):
    if file_name.endswith('.csv'):
        start_time = time.time()  # Start timing

        # # Extract schema and table name
        # schema_name = file_name.split('_')[1]
        # table_name = ("_".join(file_name.split('_')[2:])).split('.')[0]

        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)

        # Create a DataProfile object
        data_profiler = Profiler.DataProfile(file_path=file_path, env_name='prod', schema_name='schema_name',
                                             table_name='table_name', output_folder='output', skip_col_stats='N',
                                             skip_table_stats='N')
        data_profiler.first_phase()
        data_profiler.second_phase()
        data_profiler.third_phase()
        data_profiler.fourth_phase()
        fifth_phase_result = data_profiler.fifth_phase()

        # Generate HTML report and calculate its size
        report_path = data_profiler.generate_html_report()  # Assuming it returns the path of the generated report
        report_size_bytes = os.path.getsize(report_path)
        report_size_kb = report_size_bytes / 1024
        report_size_mb = report_size_kb / 1024
        report_size_gb = report_size_mb / 1024

        # Measure elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Get number of rows and columns
        row_count = len(data_profiler.df)
        column_count = len(data_profiler.df.columns)
        print(
            f"\nDone {file_path} \nRows: {row_count} \nColumns: {column_count} \nTime Taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Report Size: {report_size_gb:.2f} GB / {report_size_mb:.2f} MB / {report_size_kb:.2f} KB\n")
