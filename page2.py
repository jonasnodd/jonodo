import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import re
import numpy as np
from matplotlib.pyplot import cm
from dateutil.parser import parse
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64

# Constants
MEAN_RANGE_MIN = 1
MEAN_RANGE_MAX = 1000
DEFAULT_MEAN_RANGE = 400
BATCH_START_DEFAULT = 100
BATCH_STOP_DEFAULT = 1000

# Function to create a natural sort key
def natural_sort_key(s):
    return tuple(int(part) if re.match(r'^\d+$', part) else part for part in re.split(r'(\d+)', s))

# Function to find the indexes of local minima in a list
def find_local_minima_indexes(input_list):
    local_minima_indexes = []
    
    # Check if the list is empty or contains only one element
    if len(input_list) < 2:
        return local_minima_indexes
    
    for i in range(1, len(input_list) - 1):
        if input_list[i] < input_list[i - 1] and input_list[i] < input_list[i + 1] and input_list[i]<=max(input_list)/10:
            local_minima_indexes.append(i)
    
    return local_minima_indexes

def load_dataframes_from_files(uploaded_files):
    dfs = []
    
    for uploaded_file in uploaded_files:
        # Check the file type and read accordingly
        if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # XLSX file
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading XLSX file {uploaded_file.name}: {e}")
                continue
        else:
            # JSON file
            try:
                data = json.load(uploaded_file)
            except Exception as e:
                st.error(f"Error reading JSON content from file {uploaded_file.name}: {e}")
                continue
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(data)

        # Append the DataFrame to the list
        dfs.append(df)

    return dfs

def create_data_over_time_plot(combined_df, x_axis_column, y_axis, df_notime, mean_range, batch_start, batch_stop, batch_column):
    fig = go.Figure()

    if batch_column is not None:
        for column in combined_df.columns:
            if column != x_axis_column:  # Skip the x-axis column
                if column in y_axis:
                    # Plot other columns on the primary y-axis
                    fig.add_trace(go.Scatter(x=combined_df[x_axis_column], y=combined_df[column], mode='lines', name=column))

        fig.update_layout(title='Data Over Time', xaxis_title=x_axis_column, yaxis_title='Value (Primary Y-axis)')
        st.plotly_chart(fig)

def create_template_batch_plot(combined_df, x_axis_column, batch_start, batch_stop, batch_column):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=combined_df[x_axis_column][batch_start:batch_stop], y=combined_df[batch_column][batch_start:batch_stop], mode='lines'))
    fig.update_layout(title='Template Batch', xaxis_title=x_axis_column, yaxis_title='Value (Primary Y-axis)')
    st.plotly_chart(fig)

def create_all_batches_plot(df_notime, batch_start, batch_stop, batch_column, mean_range, y_axis):
    abssum = []
    rolling_mean = df_notime.rolling(mean_range, center=True).mean()
    
    for i in range(-batch_start, len(df_notime) - batch_stop):
        temp_batch = pd.Series(rolling_mean[batch_column][batch_start + i:batch_stop + i].values.flatten())
        template_batch = pd.Series(rolling_mean[batch_column][batch_start:batch_stop].values.flatten())
        abssum.append(template_batch.subtract(temp_batch).abs().mean())

    x = np.linspace(0, batch_stop - batch_start, batch_stop - batch_start)
    indexes = find_local_minima_indexes(abssum)

    fig = go.Figure()
    
    selected_batches = st.multiselect("Select batches to plot", options=list(range(1, len(indexes) + 1)), default=list(range(1, len(indexes) + 1)))

    for index in selected_batches:
        for ele in y_axis:
            if ele == batch_column:
                fig.add_trace(go.Scatter(x=x, y=df_notime[ele][indexes[index - 1]:indexes[index - 1] + batch_stop - batch_start], 
                                         mode='lines', name=f'{ele} for batchnr.:{index}'))
            else:
                fig.add_trace(go.Scatter(x=x, y=df_notime[ele][indexes[index - 1]:indexes[index - 1] + batch_stop - batch_start], 
                                         mode='lines', name=f'{ele} for batchnr.:{index}'))

    fig.update_layout(title='Selected batches in dataset', xaxis_title='Timestamps for samples', yaxis_title='Value (Primary Y-axis)')
    return fig

def app():
    st.title("Page 2 - Batch Analysis")
    st.write('Here you can addmultiple data in a time series and divide it into similar batches')
    uploaded_files = st.file_uploader("Choose multiple files", accept_multiple_files=True)

    if uploaded_files:
        dfs = load_dataframes_from_files(uploaded_files)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            x_axis_column = st.selectbox("Select X-axis column", combined_df.columns)
            combined_df[x_axis_column] = pd.to_datetime(combined_df[x_axis_column])
            combined_df.sort_values(by=x_axis_column, inplace=True)
            combined_df.reset_index(drop=True,inplace=True)

            st.success("DataFrames created and combined successfully!")
            st.dataframe(combined_df)

            df_notime = combined_df[combined_df.columns[~combined_df.columns.isin([x_axis_column])]]
            y_axis = st.multiselect('Select Y-data to analyze', options=combined_df.columns, default=combined_df.columns[1])

            mean_range = st.number_input('Insert mean range (lower values give more sensitivity)', 
                                         min_value=MEAN_RANGE_MIN, max_value=len(combined_df)//2, value=DEFAULT_MEAN_RANGE)

            batch_start = st.number_input("Enter batch start", min_value=1,value=BATCH_START_DEFAULT)
            batch_stop = st.number_input("Enter batch stop", max_value=len(combined_df)-2, value=BATCH_STOP_DEFAULT)
            batch_column = st.selectbox('Select Y-data to split into batches', options=y_axis, index=0)

            create_data_over_time_plot(combined_df, x_axis_column, y_axis, df_notime, mean_range, batch_start, batch_stop, batch_column)
            create_template_batch_plot(combined_df, x_axis_column, batch_start, batch_stop, batch_column)
            fig = create_all_batches_plot(df_notime, batch_start, batch_stop, batch_column, mean_range, y_axis)
            st.plotly_chart(fig)

            mybuff = StringIO()
            fig.write_html(mybuff, include_plotlyjs='cdn')
            mybuff = BytesIO(mybuff.getvalue().encode())
            b64 = base64.b64encode(mybuff.read()).decode()
            href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download plot</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    app()