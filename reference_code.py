import datetime

start_time = datetime.datetime.now()
import os
import glob
import csv
import time
import subprocess
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
import pyspark.dbutils
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum

spark = SparkSession.builder.getOrCreate()
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_dbutils(spark):
    try:
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
    except ImportError:
        import IPython
        dbutils = IPython.get_ipython().user_ns["dbutils"]
    return dbutils


def call_delta(schema_name, table_name):
    filename = "dbfs:/mnt/cdl/prd/" + schema_name + "/" + table_name
    file_type = "delta"
    infer_schema = "false"
    first_row_is_header = "True"
    delimiter = ","

    df = spark.read.format(file_type) \
        .option("inferSchema", infer_schema) \
        .option("header", first_row_is_header) \
        .option("sep", delimiter) \
        .load(filename)
    df.show()
    dbutils = get_dbutils(spark)
    dbutils.fs.rm("/delta_table", True)
    df.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("delta_table/")


def final_report(envirn_name, schema_name, table_name):
    # table_1 started
    environment = envirn_name
    schema = schema_name
    table = table_name
    today = datetime.datetime.now()
    times = today.strftime('%m-%d-%Y_%H-%M_%p')
    times1 = today.strftime('%m-%d-%Y')

    call_delta(schema, table)
    dbutils = get_dbutils(spark)

    def get_csv_files(directory_path):
        """recursively list path of all csv files in path directory """
        csv_files = []
        files_to_treat = dbutils.fs.ls(directory_path)
        while files_to_treat:
            path = files_to_treat.pop(0).path
            if path.endswith('/'):
                files_to_treat += dbutils.fs.ls(path)
            elif path.endswith('.csv'):
                csv_files.append(path)

        return csv_files

    csv_files = get_csv_files("/delta_table/")
    db_path = str(' '.join(csv_files))
    local_path = "D:/Users/APati116/PycharmProjects/db_connect_demo/spark_connect_files/" + schema + "_" + table + "_" + times1 + ".csv"
    os.system("databricks fs cp " + db_path + ' ' + str(local_path))

    df = pd.read_csv(
        'D:/Users/APati116/PycharmProjects/db_connect_demo/spark_connect_files/' + schema + "_" + table + "_" + times1 + ".csv")
    data_count = len(df)
    colm_count = len(df.columns)
    size_used = df.memory_usage(deep=True).sum()  # (gives results in bytes)

    def convert_bytes(size):
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return "%3.1f %s" % (size, x)
            size /= 1024.0
        return size

    def my_value(number):
        return ("{:,}".format(number))

    data = {
        'Environment': [environment],
        'Schema Name': [schema],
        'Table Name': [table],
        'Date/Time': [times],
        'Total Row Count': [my_value(data_count)],
        'Total Col Count': [colm_count],
        'Data Volume': convert_bytes(size_used)  # dataframe memory usage
    }

    df1 = pd.DataFrame(data)
    df1.to_csv('spark_files/csv_files/report.csv', index=False)

    # table_2_started
    Colm_Name = []
    Data_Type = []
    Total_Rows = []
    Null_Rows = []
    Null_Rows_Percent = []
    Non_Null_Rows = []
    Non_Null_Rows_Percent = []
    Distinct_Values = []
    uniqueness_index = []
    array_median = []
    median = []
    min_val = []
    max_val = []
    num_precision = []
    num_scale = []
    actual_num_prescision = []
    actual = []
    no_of_occur = []
    val = []
    for j in df.columns:
        Colm_Name.append(j)
        Data_Type.append(df[j].dtype)
        Total_Rows.append(len(df[j]))
        Null_Rows.append(df[j].isnull().sum())
        Null_Rows_Percent.append((100 * df[j].isnull().sum()) / (len(df[j])))
        Non_Null_Rows.append((len(df[j])) - (df[j].isnull().sum()))
        Non_Null_Rows_Percent.append((100 * (len(df[j]) - df[j].isnull().sum())) / (len(df[j])))
        Distinct_Values.append(df[j].nunique())
        uniqueness_index.append(np.format_float_positional(df[j].nunique() / len(df[j]), trim='-', precision=5))
        d = dict(list(dict(df[j].value_counts()).items())[0:10])
        s1 = []
        s2 = []
        for k, v in d.items():
            s1.append(str(k))
            s2.append(str(v))
        sr = ''
        for i in range(len(s1)):
            if i == len(s1) - 1:
                sr += s1[i] + " (" + s2[i] + ")"
            else:
                sr += s1[i] + " (" + s2[i] + ")" + "," + " "

        no_of_occur.append(sr)
        if df[j].dtype == 'int64':
            df[j] = df[j].replace(np.nan, 0)
            min_val.append(df[j].min())
            max_val.append(df[j].max())

            for i in df[j]:
                if i != 'nan':
                    actual.append(i)
            actual_num_prescision.append(max(actual))
            actual.clear()

            for i in df[j]:
                if i != 'nan':
                    array_median.append([i])
            median.append(np.median(array_median))
            array_median.clear()

        if df[j].dtype == 'float64':
            df[j] = df[j].replace(np.nan, 0)
            for i in df[j]:
                if i != 'nan':
                    array_median.append([i])
            median.append(np.median(array_median))
            array_median.clear()
            min_val.append(df[j].min())
            max_val.append(df[j].max())

            for i in df[j]:
                if isfloat(i) and str(i) != 'nan':
                    actual.append(i)
            actual_num_prescision.append(max(actual))
            actual.clear()

        if df[j].dtype == 'object':
            median.append('NA')
            actual_num_prescision.append('NA')
            for i in df[j].astype(str):
                if i != 'nan':
                    val.append(i)
            max_val.append(max(val))
            min_val.append(min(val))
            val.clear()

    col_seq = range(1, len(Colm_Name) + 1)
    Null_Rows_Per = ['%.2f' % elem for elem in Null_Rows_Percent]
    Non_Null_Rows_Per = ['%.2f' % elem for elem in Non_Null_Rows_Percent]
    num_precision = []
    num_scale = []
    for i in actual_num_prescision:
        if type(i) == float:
            m = str(i)
            num_precision.append(len(m) - 1)
            if '.' in m:
                idx = m.index('.')
                num_scale.append(len(m[idx + 1:]))
        elif type(i) == int:
            p = str(i)
            num_precision.append(len(p))
            num_scale.append("NA")
        else:
            num_precision.append(i)
            num_scale.append(i)
    prescale = list(map(lambda x, y: str(x) + ', ' + str(y), num_precision, num_scale))

    data = {
        'Column Name': Colm_Name,
        'Column Data Type': Data_Type,
        'NULL Record Count': Null_Rows,
        'Percentage of NULL Values(%)': Null_Rows_Per,
        'NOT NULL Record Count': Non_Null_Rows,
        'Percentage of NOT NULL Values(%)': Non_Null_Rows_Per,
        'Number of Distinct Values in the field': Distinct_Values,
        'Uniqueness Index (Number of Distinct Values/ Total Number of Records)': uniqueness_index,
        'Median for Numeric Data Type': median,
        'Maximum Available Precision Value for Numeric Data Type': actual_num_prescision,
        'Precision, Scale Value for Numeric Data Type': prescale
    }
    df2 = pd.DataFrame(data, index=col_seq)
    df2.index.name = 'Column Sequence'
    df2.to_csv('spark_files/csv_files/table.csv')

    # Summary_table

    mpg = dict(df2.groupby(['Column Data Type']).size())

    def get_val(keys):
        for key, value in mpg.items():
            if keys == key:
                return value
        return "NA"

    singlecount = data_count
    nullrecd = int(df2['NULL Record Count'].sum())
    notnullrecd = int(df2['NOT NULL Record Count'].sum())
    noofrows = len(df2['Column Name'])
    dtcount = len(df2['Column Data Type'].unique())
    intcount = get_val('int64')
    floatcount = get_val('float64')
    objectcount = get_val('O')
    datecolcount = get_val('date')
    timestampcolcount = get_val('timestamp')
    doublecolcount = get_val('Double')
    len_df = len(df)
    dat = {
        "Total Row Count": [my_value(singlecount)],
        'Total Column Count': [noofrows],
        'Total Data Type Count ': [dtcount],
        'Int Column Count': [intcount],
        'Float Column Count': [floatcount],
        'Object Column Count': [objectcount],
        'Date Column Count': [datecolcount],
        'TIMESTAMP Column Count': [timestampcolcount],
        'Double Column Count': [doublecolcount],
    }
    df_sum = pd.DataFrame(dat)
    df_sum.to_csv('spark_files/csv_files/summarytable.csv', index=False)

    # table_4 (occurrence_table)

    df5 = {'Column Name': Colm_Name,
           'Minimum Value': min_val,
           'Maximum Value': max_val,
           'Top 10 values along with number of occurrence  [value(occurrence)]': no_of_occur}
    df6 = pd.DataFrame(df5, index=col_seq)
    df6.to_csv('spark_files/csv_files/occurrence.csv')

    ##########GRAPHS##############
    ###GRAPHS1
    result = df2.groupby(['Column Data Type']).size()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    x = np.arange(len(result))  # the label locations
    width = 0.20  # the width of the bars
    rects1 = ax.bar(result, width, color=['firebrick', 'green', 'blue'])
    plt.rcParams.update({'font.size': 12})
    rects = ax.patches
    # plot the result
    plots = sns.barplot(x=result.index, y=result.values)
    for bar in plots.patches:
        # Using Matplotlib's annotate function and
        # passing the coordinates where the annotation shall be done
        # x-coordinate: bar.get_x() + bar.get_width() / 2
        # y-coordinate: bar.get_height()
        # free space to be left to make graph pleasing: (0, 8)
        # ha and va stand for the horizontal and vertical alignment
        plots.annotate(format(bar.get_height(), '.0f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=18, xytext=(0, 8),
                       textcoords='offset points')
    ax.text(0, 1.08, 'Total Data Type Count',
            transform=ax.transAxes, size=20, weight=600, ha='left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', colors='black', labelsize=20)
    ax.tick_params(axis='y', colors='black', labelsize=16)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.grid(color='black', axis='y', ls='-', lw=0.25)
    # plt.tick_params(axis='y', which='both', left=False, top=False, labelbottom=False)
    ax.set_axisbelow(True)
    ax.set(xlabel=None)
    plt.savefig('spark_files/images/image_1.png')

    class PDF(FPDF):

        def footer(self):
            # Setting position at 1.5 cm from bottom:
            self.set_y(-15)
            # Setting font: helvetica italic 8
            self.set_font("helvetica", "I", 8)
            # Setting text color to gray:
            self.set_text_color(128)
            # Printing page number
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

        def basic_report(self, headings, rows):

            for heading in headings:
                self.set_fill_color(255, 100, 0)
                self.set_text_color(255)
                self.set_draw_color(255, 0, 0)
                self.set_line_width(0.3)
                self.set_font('helvetica', 'B', 12)
                self.cell(col_width1, 2 * th, heading, align='C', border=1, fill=True)
            self.ln(2 * th)
            for row in rows:
                for col in row:
                    self.set_fill_color(224, 235, 255)
                    self.set_text_color(0)
                    self.set_font('helvetica', '', 12)
                    fill = False
                    self.cell(col_width1, 2 * th, col, align='C', border=1, fill=fill)
                self.ln(2 * th)

        def graph(self, image_path1):
            self.image(image_path1, x=30, y=60, w=110)
            self.set_font("Arial", size=12)
            self.ln(8 * th)  # move 85 down

        def summary_table(self, data):
            lh_list = []  # list with proper line_height for each row
            use_default_height = 0  # flag
            line_height = th * 2
            for row in data:
                for datum in row:
                    word_list = datum.split()
                    number_of_words = len(word_list)  # how many words
                    if number_of_words > 3:  # names and cities formed by 2 words like Los Angeles are ok)
                        use_default_height = th
                        new_line_height = self.font_size * (number_of_words / 3)  # new height change according to data
                if not use_default_height:
                    lh_list.append(round(line_height))
                else:
                    lh_list.append(round(new_line_height))
                    use_default_height = 0
            for i, row in enumerate(data):
                for j, datum in enumerate(row):
                    if j == 0:
                        if datum == "Total Row Count":
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='L',
                                            ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = 10
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=False)

                    if j == 1:

                        if datum == 'Total Column Count':
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='C',
                                            ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = 10

                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            fill = False
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 2:
                        if datum == 'Total Data Type Count':
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='C',
                                            ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = 10
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            fill = False
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 3:

                        if datum == 'Int Column Count':
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='C',
                                            ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = 10
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            fill = False
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 4:

                        if datum == 'Float Column Count':
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='C',
                                            ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = 10
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            fill = False
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 5:

                        if datum == 'Object Column Count':
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='C',
                                            ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = 10
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            fill = False
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 6:

                        if datum == "Date Column Count":
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = 10
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            fill = False
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 7:

                        if datum == "TIMESTAMP Column Count":
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = 10
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            fill = False
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 8:

                        if datum == "Double Column Count":
                            line_height = 20
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width5, line_height, '\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = 10
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 12)
                            fill = False
                            self.multi_cell(col_width5, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                self.ln(line_height)
            self.ln(2 * th)

        def occur_table(self, headings, rows):
            def render_table_header():
                self.set_fill_color(255, 100, 0)
                self.set_text_color(255)
                self.set_draw_color(255, 0, 0)
                self.set_line_width(0.3)
                self.set_font('helvetica', 'B', 12)
                for i, heading in enumerate(headings):
                    if i == 0:
                        self.cell(col_width3 / 5, 2 * th, 'Sr. No.', align='C', border=1, fill=True)
                    if i == 1:
                        self.cell(col_width3 / 1.3, 2 * th, heading, align='L', border=1, fill=True)
                    if i == 2:
                        self.cell(col_width3 / 2, 2 * th, heading, align='L', border=1, fill=True)
                    if i == 3:
                        self.cell(col_width3 / 2, 2 * th, heading, align='L', border=1, fill=True)
                    if i == 4:
                        self.cell(col_width3 * 3, 2 * th, heading, align='L', border=1, fill=True)
                self.ln(2 * th)
                pdf.set_font(style="")

            lh_list = []  # list with proper line_height for each row
            use_default_height = 0  # flag
            for row in rows:
                line_height = th * 2
                for datum in row:
                    word_list = datum.split()
                    number_of_words = len(word_list)  # how many words
                    if number_of_words > 8:  # names and cities formed by 2 words like Los Angeles are ok)
                        use_default_height = th
                        new_line_height = pdf.font_size * (
                                number_of_words / 8)  # new height change according to data
                if not use_default_height:
                    lh_list.append(round(line_height))
                else:
                    lh_list.append(round(new_line_height))
                    use_default_height = 0
            li = []
            for m in lh_list:
                if m < 2 * th:
                    g = round(2 * th)
                    li.append(g + 2)
                else:
                    li.append(m + 2)

            render_table_header()
            for j, row in enumerate(rows):
                if self.will_page_break(line_height):
                    self.add_page()
                    render_table_header()

                for flag, datum in enumerate(row):
                    self.set_fill_color(224, 235, 255)
                    self.set_text_color(0)
                    self.set_font('helvetica', '', 10)
                    fill = False
                    line_height = 1 + li[j] + 1  # choose right height for current row
                    datum = '\n' + datum + '\n'
                    if flag == 0:
                        self.multi_cell(col_width3 / 5, line_height, datum, border=1, align='C', ln=3,
                                        max_line_height=self.font_size, fill=fill)
                    elif flag == 1:
                        self.multi_cell(col_width3 / 1.3, line_height, datum, border=1, align='L', ln=3,
                                        max_line_height=self.font_size, fill=fill)
                    elif flag == 2:
                        self.multi_cell(col_width3 / 2, line_height, datum, border=1, align='L', ln=3,
                                        max_line_height=self.font_size, fill=fill)
                    elif flag == 3:
                        self.multi_cell(col_width3 / 2, line_height, datum, border=1, align='L', ln=3,
                                        max_line_height=self.font_size, fill=fill)
                    elif flag == 4:
                        self.multi_cell(col_width3 * 3, line_height, datum, border=1, align='L', ln=3,
                                        max_line_height=self.font_size, fill=fill)
                    flag = flag + 1
                self.ln(line_height)
            self.ln(2 * th)

        def basic_table(self, data):
            lh_list = []  # list with proper line_height for each row
            use_default_height = 0  # flag
            line_height = th * 2
            for row in data:
                for datum in row:
                    word_list = datum.split()
                    number_of_words = len(word_list)  # how many words
                    if number_of_words > 3:  # names and cities formed by 2 words like Los Angeles are ok)
                        use_default_height = th
                        new_line_height = self.font_size * (number_of_words / 3)  # new height change according to data
                if not use_default_height:
                    lh_list.append(round(line_height))
                else:
                    lh_list.append(round(new_line_height))
                    use_default_height = 0

            for i, row in enumerate(data):

                for j, datum in enumerate(row):
                    if j == 0:
                        if datum == 'Column Sequence':
                            line_height = 30
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = 7
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=False)

                    if j == 1:

                        if datum == 'Column Name':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4 * 2, line_height, '\n\n' + datum + '\n\n', border=1, align='L',
                                            ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4 * 2, line_height, datum, border=1, align='L', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 2:

                        if datum == 'Column Data Type':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 3:

                        if datum == 'NULL Record Count':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '0':
                            line_height = lh_list[j]
                            self.set_fill_color(255, 255, 0)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == str(data_count):
                            line_height = lh_list[j]
                            # self.set_fill_color(255, 255, 0)
                            pdf.set_fill_color(r=255, g=0, b=255)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 4:
                        if datum == 'Percentage of NULL Values(%)':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '0.00':
                            line_height = lh_list[j]
                            self.set_fill_color(255, 255, 0)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '100.00':
                            line_height = lh_list[j]
                            self.set_fill_color(69, 171, 82)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 5:
                        if datum == 'NOT NULL Record Count':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '0':
                            line_height = lh_list[j]
                            self.set_fill_color(255, 255, 0)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == str(data_count):
                            line_height = lh_list[j]
                            # self.set_fill_color(255, 255, 0)
                            pdf.set_fill_color(r=255, g=0, b=255)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 6:
                        if datum == 'Percentage of NOT NULL Values(%)':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '0.00':
                            line_height = lh_list[j]
                            self.set_fill_color(255, 255, 0)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '100.00':
                            line_height = lh_list[j]
                            self.set_fill_color(69, 171, 82)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)

                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 7:
                        if datum == 'Number of Distinct Values in the field':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif int(datum) == 1:
                            line_height = lh_list[j]
                            # self.set_fill_color(255, 255, 0)
                            pdf.set_fill_color(r=10, g=30, b=255)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '0':
                            line_height = lh_list[j]
                            self.set_fill_color(255, 255, 0)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == str(data_count):
                            line_height = lh_list[j]
                            # self.set_fill_color(255, 255, 0)
                            pdf.set_fill_color(r=255, g=0, b=255)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 8:
                        if datum == 'Uniqueness Index (Number of Distinct Values/ Total Number of Records)':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 11)
                            self.multi_cell(col_width4, line_height, '\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)

                        elif float(datum) > 0.5 and float(datum) < 1:
                            line_height = lh_list[j]
                            # self.set_fill_color(69, 171, 82)
                            pdf.set_fill_color(r=165, g=10, b=255)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif float(datum) == 1:
                            line_height = lh_list[j]
                            # self.set_fill_color(69, 171, 82)
                            self.set_fill_color(255, 185, 15)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif float(datum) < 0.5 and float(datum) > 0:
                            line_height = lh_list[j]
                            # self.set_fill_color(255, 255, 0)
                            self.set_fill_color(211, 211, 211)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '0':
                            line_height = lh_list[j]
                            self.set_fill_color(255, 255, 0)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 9:
                        if datum == 'Median for Numeric Data Type':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '0.0':
                            line_height = lh_list[j]
                            self.set_fill_color(255, 255, 0)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == 'NA':
                            line_height = lh_list[j]
                            self.set_fill_color(0, 216, 230)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 10:
                        if datum == 'Maximum Available Precision Value for Numeric Data Type':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n' + datum + '\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == '0.0':
                            line_height = lh_list[j]
                            self.set_fill_color(255, 255, 0)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == 'NA':
                            line_height = lh_list[j]
                            self.set_fill_color(0, 216, 230)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 11:
                        if datum == 'Precision, Scale Value for Numeric Data Type':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == 'NA, NA':
                            line_height = lh_list[j]
                            self.set_fill_color(0, 216, 230)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)

                    if j == 12:
                        if datum == 'Maximum Available Scale Value for Numeric Data Type':
                            lh_list[0] = 30
                            line_height = lh_list[0]
                            self.set_fill_color(255, 100, 0)
                            self.set_text_color(255)
                            self.set_draw_color(255, 0, 0)
                            self.set_line_width(0.3)
                            self.set_font('helvetica', 'B', 12)
                            self.multi_cell(col_width4, line_height, '\n\n' + datum + '\n\n', border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        elif datum == 'NA':
                            line_height = lh_list[j]
                            self.set_fill_color(0, 216, 230)
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=True)
                        else:
                            line_height = lh_list[j]
                            self.set_fill_color(224, 235, 255)
                            self.set_text_color(0)
                            self.set_font('helvetica', '', 10)
                            fill = False
                            self.multi_cell(col_width4, line_height, datum, border=1, align='C', ln=3,
                                            max_line_height=self.font_size, fill=fill)
                self.ln(line_height)
            self.ln(2 * th)

        def performace_cell(self, perf):
            self.set_fill_color(255, 100, 0)
            self.set_text_color(255)
            self.set_draw_color(255, 0, 0)
            self.set_line_width(0.3)
            self.set_font('helvetica', 'B', 14)
            self.cell(epw / 4, 3 * th, perf + '   HH:MM:SS.ssss', align='C', border=1, fill=True)
            self.ln(3 * th)

        def legend(self):
            self.ln(2 * th)
            self.set_fill_color(255, 255, 0)
            self.set_font('helvetica', 'B', 10)
            self.cell(col_width6, 2 * th, 'Value = 0 or 0.00', align='C', border=1, fill=True)

            self.set_fill_color(69, 171, 82)
            self.set_font('helvetica', 'B', 10)
            self.cell(col_width6, 2 * th, 'Value = 100.00', align='C', border=1, fill=True)

            pdf.set_fill_color(r=255, g=0, b=255)
            self.set_font('helvetica', 'B', 10)
            self.cell(col_width6, 2 * th, 'Total Row Count', align='C', border=1, fill=True)

            pdf.set_fill_color(r=10, g=30, b=255)
            self.set_font('helvetica', 'B', 10)
            self.cell(col_width6, 2 * th, 'Distinct Values = 1', align='C', border=1, fill=True)

            self.set_fill_color(211, 211, 211)
            self.set_font('helvetica', 'B', 10)
            self.cell(col_width6, 2 * th, ' 0 > Uniqueness Index < 0.5', align='C', border=1, fill=True)

            pdf.set_fill_color(r=165, g=10, b=255)
            self.set_font('helvetica', 'B', 10)
            self.cell(col_width6, 2 * th, '1 > Uniqueness Index > 0.5', align='C', border=1, fill=True)

            self.set_fill_color(255, 185, 15)
            self.set_font('helvetica', 'B', 10)
            self.cell(col_width6, 2 * th, 'Uniqueness Index = 1', align='C', border=1, fill=True)

            self.set_fill_color(0, 216, 230)
            self.set_font('helvetica', 'B', 10)
            self.cell(col_width6, 2 * th, 'NA', align='C', border=1, fill=True)
            self.ln(2 * th)

    def load_data_from_csv(csv_filepath):
        headings, rows = [], []
        with open(csv_filepath, encoding="utf8") as csv_file:
            for row in csv.reader(csv_file, delimiter=","):
                if not headings:  # extracting column names from first row:
                    headings = row
                else:
                    rows.append(row)
        return headings, rows

    col_names, data = load_data_from_csv("spark_files/csv_files/report.csv")
    # pdf = PDF()
    pdf = PDF('L', 'mm', 'A3')
    pdf.add_page()

    pdf.set_font("helvetica", size=8)
    epw = pdf.w - 2 * pdf.l_margin
    col_width1 = epw / 7
    col_width5 = epw / 9
    col_width2 = epw / 4
    col_width3 = epw / 5
    col_width4 = epw / 13
    col_width6 = epw / 8
    col_width7 = epw / 2
    pdf.ln(5)
    pdf.set_font('helvetica', 'BU', 16.0)
    pdf.cell(epw, 0.0, 'Data Profiling Report', align='C')
    pdf.set_font('helvetica', '', 10.0)
    pdf.ln(8)

    pdf.set_font('helvetica', 'BU', 14.0)
    pdf.cell(epw, 0.0, 'Report Generated For :-', align='L')
    pdf.set_font('helvetica', '', 10.0)
    pdf.ln(8)

    th = pdf.font_size

    pdf.basic_report(col_names, data)
    pdf.ln(4 * th)

    pdf.set_font('helvetica', 'BU', 14.0)
    pdf.cell(epw, 0.0, 'Graph :-', align='L')
    pdf.set_font('helvetica', '', 10.0)
    # pdf.ln(7)
    pdf.ln(11 * th)
    img1 = 'spark_files/images/image_1.png'

    pdf.graph(img1)

    pdf.ln(8 * th)
    pdf.ln(6)
    col_names, data = load_data_from_csv("spark_files/csv_files/summarytable.csv")
    data.insert(0, col_names)
    tuple_data = tuple([tuple(l) for l in data])

    pdf.set_font("helvetica", size=8)
    pdf.set_font('helvetica', 'BU', 14.0)
    pdf.cell(epw, 0.0, 'Summary Table :-', align='L')
    pdf.set_font('helvetica', '', 10.0)
    pdf.ln(8)
    pdf.summary_table(tuple_data)
    pdf.ln(3 * th)

    col_names, data = load_data_from_csv("spark_files/csv_files/occurrence.csv")
    # data.insert(0, col_names)
    # tuple_data = tuple([tuple(l) for l in data])

    pdf.set_font("helvetica", size=8)
    pdf.set_font('helvetica', 'BU', 14.0)
    pdf.cell(epw, 0.0, 'Minimum, Maximum and Top 10 values along with number of occurrence :-', align='L')
    pdf.set_font('helvetica', '', 10.0)
    pdf.ln(8)
    # pdf.occur_table(tuple_data)
    pdf.occur_table(col_names, data)
    pdf.ln(4 * th)

    col_names, data = load_data_from_csv("spark_files/csv_files/table.csv")
    data.insert(0, col_names)
    tuple_data = tuple([tuple(l) for l in data])

    pdf.set_font("helvetica", size=8)
    pdf.set_font('helvetica', 'BU', 14.0)
    pdf.cell(epw, 0.0, 'Data Profilling Statistics :-', align='L')
    pdf.set_font('helvetica', '', 10.0)
    pdf.ln(8)
    pdf.basic_table(tuple_data)
    pdf.ln(th)

    pdf.set_font("helvetica", size=8)
    pdf.set_font('helvetica', 'BU', 12.0)
    pdf.cell(epw, 0.0, 'Color Legend :-', align='L')
    pdf.legend()
    pdf.ln(2 * th)

    pdf.ln(8)
    pdf.set_font("helvetica", size=8)
    pdf.set_font('helvetica', 'BU', 14.0)
    pdf.cell(epw, 0.0, 'Data Profilling Processing Time :-', align='L')
    pdf.ln(8)

    end_time = datetime.datetime.now()
    code_process_time = str(end_time - start_time)
    print("*******************Code Is Working************************")
    print('Data Profilling Processing Time :' + str(code_process_time))
    print("**********************************************************")

    pdf.set_font('helvetica', 'BU', 12.0)
    pdf.performace_cell(code_process_time)
    x = 'spark_files/pdf/' + str(schema) + '_' + str(table) + '_' + str(times) + '.pdf'
    pdf.output(name=x, dest='F')


def open_csv_file(csv_filepath_1, csv_filepath_2):
    headings1, rows1 = [], []
    with open(csv_filepath_1, encoding="utf8") as csv_file:
        for row in csv.reader(csv_file, delimiter=","):
            if not headings1:  # extracting column names from first row:
                headings1 = row
            else:
                rows1.append(row)
    headings2, rows2 = [], []
    with open(csv_filepath_2, encoding="utf8") as csv_file:
        for row in csv.reader(csv_file, delimiter=","):
            if not headings2:  # extracting column names from first row:
                headings2 = row
            else:
                rows2.append(row)
    return headings1, rows1, headings2, rows2


headings1, rows1, headings2, rows2 = open_csv_file('schema_table.csv', 'jnj_credentials.csv')

li = []
for i in rows2:
    li.append(i[1])

host_id = li[0]
token = li[1]
cluster_id = li[2]
org_id = li[3]
port = li[4]

stdin_list = [host_id, token, cluster_id, org_id, port]
stdin_string = '\n'.join(stdin_list)
try:
    output = subprocess.run('databricks-connect configure', text=True, input=stdin_string, shell=True)
    print(output)
    print('=========Databricks_Connected==============')
except:
    print('not connected')

for i in range(len(rows1)):
    env = rows1[i][0]
    schema = rows1[i][1]
    table = rows1[i][2]
    final_report(env, schema, table)
######################### for local csv files

# final_report('devlopment', 'lpfg_core', 'delv_line') ##Y
# final_report('devlopment', 'lpfg_core', 'eur_inv_bal') ##Y
# final_report('devlopment', 'lpfg_core', 'em_inv_report') ##Y
# final_report('devlopment', 'lpfg_core', 'inv_grc_matcost') ##Y
# final_report('devlopment', 'lpfg_core', 'bom_hdr') ##W
# final_report('devlopment', 'lpfg_core', 'matl_inv') ##W
# final_report('devlopment', 'lpfg_core', 'Auto') ##Y

######################### for databricks csv files

# final_report('devlopment', 'lpfg_core', 'em_inv_report') ##Y
# final_report('devlopment', 'lpfg_wrk', 'inv_grc_matcost') ##Y
# final_report('devlopment', 'lpfg_core', 'bom_hdr') ##W
# final_report('devlopment', 'lpfg_core', 'delv_line') ##Y
# final_report('devlopment', 'lpfg_wrk', 'eur_inv_bal') ##Y


######################### for larger tables
# final_report('Production', 'stf', 'aa')   #small file
# final_report('Production', 'bpi', 'hpo')    #small file
# final_report('Production', 'stf', 'mseg')  #large file
# final_report('Production', 'stf', 'ekko')  # large file
# final_report('Production', 'atl', 'afpo')  # large file
# final_report('Production', 'SC_CORE', 'SalaryTable')
# final_report('Production', 'LPFG_CORE', 'jnj_spark_file')
# final_report('Production', 'SC_CORE', 'Volume_Report_Extract')pdf_file
# final_report('Production', 'SC_CORE', 'part-00000-tid-6327992309494733601-5f23a3f7-0a1f-4d39-9889-17a67e198533-7762-1-c000') ##W
