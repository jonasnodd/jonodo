import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import numpy as np
from matplotlib.pyplot import cm
from dateutil.parser import parse

# Constants
MEAN_RANGE_MIN = 1
MEAN_RANGE_MAX = 1000
DEFAULT_MEAN_RANGE = 400
BATCH_START_DEFAULT = 100
BATCH_STOP_DEFAULT = 1000

# Function to create a natural sort key
def natural_sort_key(s):
    return tuple(int(part) if re.match(r'^\d+$', part) else part for part in re.split(r'(\d+)', s))

def is_datetime_column(column):
    # Heuristic to check if the column name contains common datetime keywords
    datetime_keywords = ["date", "time", "dt"]
    for keyword in datetime_keywords:
        if keyword in column.lower():
            return True

    # Heuristic to check if the column values can be parsed as datetime
    try:
        df = pd.DataFrame({column: ["2023-01-01 12:00:00"]})
        df[column] = pd.to_datetime(df[column])
        return True
    except Exception:
        return False

# Function to find the indexes of local minima in a list
def find_local_minima_indexes(input_list):
    local_minima_indexes = []
    
    # Check if the list is empty or contains only one element
    if len(input_list) < 2:
        return local_minima_indexes
    
    for i in range(1, len(input_list) - 1):
        if input_list[i] < input_list[i - 1] and input_list[i] < input_list[i + 1]:
            local_minima_indexes.append(i)
    
    return local_minima_indexes

def load_dataframes_from_files(uploaded_files):
    dfs = []
    
    for uploaded_file in uploaded_files:
        # Step 1: Read the uploaded file and parse JSON content
        try:
            data = json.loads(uploaded_file.read())
        except Exception as e:
            st.error(f"Error reading JSON content from file {uploaded_file.name}: {e}")
            continue

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        # Identify and convert potential datetime columns
        for column in df.columns:
            if is_datetime_column(column):
                try:
                    df[column] = pd.to_datetime(df[column]).dt.tz_convert('Europe/Oslo')
                except Exception as e:
                    pass

        # Step 2: Apply alphabetical numbering to non-numeric columns
        non_numeric_columns = df.select_dtypes(include='object').columns
        alphabetical_numbers = {value: index + 1 for index, value in enumerate(sorted(df[non_numeric_columns].stack().unique(), key=natural_sort_key))}
        df[non_numeric_columns] = df[non_numeric_columns].applymap(lambda x: alphabetical_numbers[x] if x in alphabetical_numbers else x)

        # Append the DataFrame to the list
        dfs.append(df)

    return dfs

def create_data_over_time_plot(combined_df, x_axis_column, y_axis, df_notime, mean_range, batch_start, batch_stop, batch_column):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    rolling_mean = df_notime.rolling(mean_range, center=True).mean()

    if batch_column is not None:
        for column in combined_df.columns:
            if column != x_axis_column:  # Skip the x-axis column
                if column in y_axis:
                    # Plot other columns on the primary y-axis
                    ax1.plot(combined_df[x_axis_column], combined_df[column], label=column)
                    ax1.plot(combined_df[x_axis_column], rolling_mean[column], label=column)

        ax1.set_xlabel(x_axis_column)
        ax1.set_ylabel('Value (Primary Y-axis)')
        ax1.set_title('Data Over Time')
        ax1.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

def create_template_batch_plot(combined_df, x_axis_column, batch_start, batch_stop, batch_column):
    batchplot = plt.figure(figsize=(10, 6))
    plt.plot(combined_df[x_axis_column][batch_start:batch_stop], combined_df[batch_column][batch_start:batch_stop])
    plt.title('Template Batch')
    plt.xlabel(x_axis_column)
    plt.ylabel('Value (Primary Y-axis)')
    st.pyplot(batchplot)

def create_all_batches_plot(df_notime, batch_start, batch_stop, batch_column,mean_range):
    abssum = []
    rolling_mean = df_notime.rolling(mean_range, center=True).mean()
    for i in range(-batch_start, len(df_notime) - batch_stop):
        temp_batch = pd.Series(rolling_mean[batch_column][batch_start + i:batch_stop + i].values.flatten())
        template_batch = pd.Series(rolling_mean[batch_column][batch_start:batch_stop].values.flatten())
        abssum.append(template_batch.subtract(temp_batch).abs().mean())

    x = np.linspace(0, batch_stop - batch_start, batch_stop - batch_start)
    indexes = find_local_minima_indexes(abssum)
    batchplot_all = plt.figure(figsize=(10, 6))
    color = cm.rainbow(np.linspace(0, 1, len(indexes)))

    for index in indexes:
        plt.plot(x, df_notime[batch_column][index:index + batch_stop - batch_start], label='batchnr.:' + str(indexes.index(index)), c=color[indexes.index(index)])

    plt.legend()
    plt.title('All batches in dataset')
    plt.xlabel('timestamps for samples')
    plt.ylabel('Value (Primary Y-axis)')
    st.pyplot(batchplot_all)

def app():
    st.title("Page 4 - Batch Analysis")
    uploaded_files = st.file_uploader("Choose multiple files", accept_multiple_files=True)

    if uploaded_files:
        dfs = load_dataframes_from_files(uploaded_files)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            st.success("DataFrames created and combined successfully!")
            st.dataframe(combined_df)

            df_notime = combined_df[combined_df.columns[~combined_df.columns.isin(['tid'])]]
            x_axis_column = st.selectbox("Select X-axis column", combined_df.columns)
            y_axis = st.multiselect('Select Y-data to analyze', options=combined_df.columns, default=combined_df.columns[1])

            mean_range = st.number_input('Insert mean range (lower values give more sensitivity)', 
                                         min_value=MEAN_RANGE_MIN, max_value=len(combined_df)//2, value=DEFAULT_MEAN_RANGE)

            batch_start = st.number_input("Enter batch start", min_value=1,value=BATCH_START_DEFAULT)
            batch_stop = st.number_input("Enter batch stop", max_value=len(combined_df)-2, value=BATCH_STOP_DEFAULT)
            batch_column = st.selectbox('Select Y-data to split into batches', options=y_axis, index=0)

            create_data_over_time_plot(combined_df, x_axis_column, y_axis, df_notime, mean_range, batch_start, batch_stop, batch_column)
            create_template_batch_plot(combined_df, x_axis_column, batch_start, batch_stop, batch_column)
            create_all_batches_plot(df_notime, batch_start, batch_stop, batch_column,mean_range)

if __name__ == "__main__":
    app()
