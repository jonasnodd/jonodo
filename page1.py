import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Constants
EPSILON_MIN = 0.10
EPSILON_MAX = 2.00
EPSILON_STEP = 0.01


# Function to perform natural sorting for alphanumeric strings
def natural_sort_key(s):
    return tuple(int(part) if re.match(r'^\d+$', part) else part for part in re.split(r'(\d+)', s))


# Helper function to handle non-numeric values in numeric columns for clustering
def contains_alpha(column):
    unique_values = {}
    unique_numbers = 1
    column_values = column.values

    for i in range(len(column_values)):
        value = column_values[i]
        if isinstance(value, int):
            continue
        else:
            try:
                float_value = float(value)
                column_values[i] = float_value
            except ValueError:
                if value not in unique_values:
                    unique_values[value] = None
                    column_values[i] = value

    sorted_values = sorted(unique_values.keys(), key=natural_sort_key)
    alphabetical_numbers = {value: index + 1 for index, value in enumerate(sorted_values)}

    for i in range(len(column_values)):
        value = column_values[i]
        if value in alphabetical_numbers:
            column_values[i] = alphabetical_numbers[value]

    return column_values


# Function to read data from an Excel file and preprocess it
def read_excel_file(uploaded_file, sheet_name, skip_rows_headers, skip_rows_data, cutoff_row_number):
    xls = pd.ExcelFile(uploaded_file)
    df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=skip_rows_headers)
    df = df.iloc[skip_rows_data:]
    if cutoff_row_number > 0:
        df = df.iloc[:cutoff_row_number]
    return df



# Function to preprocess the DataFrame based on user-selected data types
def preprocess_dataframe(df, selected_string_columns, selected_date_columns):
    for col in df.columns:
        if col in selected_string_columns:
            df[col] = df[col].astype(str)
        elif col in selected_date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# Function to preprocess data for clustering and return scaled features
def get_data_for_clustering(dataframe, options):
    dataframe_integerized = dataframe[options].copy()
    for option in options:
        dataframe_integerized[option] = contains_alpha(dataframe[option])

    sorted_columns = sorted(dataframe_integerized.columns, key=natural_sort_key)
    dataframe_integerized = dataframe_integerized[sorted_columns]

    features = dataframe_integerized.fillna(0).values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return dataframe_integerized, scaled_features


# Function to perform DBSCAN clustering and visualize the results
def cluster_and_visualize(dataframe_integerized, scaled_features, epsilon, focus, options):
    dbscan = DBSCAN(eps=epsilon)
    dbscan.fit(scaled_features)
    cluster_means = dataframe_integerized.groupby(dbscan.labels_).mean()
    focus_mean = list(cluster_means[focus])
    palette = sns.color_palette("bright")

    if len(focus_mean) >= 2:
        dataframe_integerized['clusters'] = dbscan.labels_
        dataframe_integerized['clusters'] = dataframe_integerized['clusters'].map(lambda x: focus_mean[x+1] if x+1 < len(focus_mean) else None)
        fig = sns.pairplot(dataframe_integerized, hue='clusters', palette=palette)
    else:
        fig = sns.pairplot(dataframe_integerized)

    return fig


# Function to visualize the correlation between selected columns
def visualize_correlation(dataframe_integerized, options):
    pearsoncorr = pd.DataFrame(dataframe_integerized[options].corr(method='pearson'))
    plot, ax = plt.subplots()
    sns.heatmap(pearsoncorr, cmap='viridis', annot=True, fmt=".1f", linewidths=0.5, ax=ax)
    ax.set_title('Pearson Correlation Factor')
    return plot


# Main application function
def app():
    st.title("Clustering")
    st.write("Here you can insert an excel-sheet with data and cluster the preferred data using DBSCAN method.")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

    if uploaded_file:
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            sheet_name = st.selectbox("Select a sheet", sheet_names, index=0)
            skip_rows_headers = st.number_input("Number of rows to skip above the column headers", min_value=0, value=0)
            skip_rows_data = st.number_input("Number of rows to skip after the column headers", min_value=0, value=0)
            cutoff_row_number = st.number_input("number of data rows", min_value=0, value=0)

            df = read_excel_file(uploaded_file, sheet_name, skip_rows_headers, skip_rows_data, cutoff_row_number)

            string_columns = df.select_dtypes(include='object').columns
            date_columns = pd.to_datetime(df.select_dtypes(include='datetime').stack(), errors='coerce').unstack().columns

            st.sidebar.write("**Column Data Types**")
            selected_string_columns = st.sidebar.multiselect("Select String columns", options=df.columns, default=list(string_columns))
            selected_date_columns = st.sidebar.multiselect("Select Date columns", options=df.columns, default=list(date_columns))

            df = preprocess_dataframe(df, selected_string_columns, selected_date_columns)

            st.success(f"DataFrame created successfully from '{uploaded_file.name}'!")
            st.dataframe(df)
            dataframe = df

            labels = np.array(dataframe.columns)

            options = st.multiselect('What do you want to trend?', labels)
            if len(options) >= 1:
                dataframe_integerized, scaled_features = get_data_for_clustering(dataframe, options)

                epsilon = st.slider('Epsilon', min_value=EPSILON_MIN, max_value=EPSILON_MAX, value=EPSILON_MAX, step=EPSILON_STEP)
                focus = st.selectbox('Which average should be presented?', options, index=0)

                fig = cluster_and_visualize(dataframe_integerized, scaled_features, epsilon, focus, options)
                st.pyplot(fig)

                plot = visualize_correlation(dataframe_integerized, options)
                st.pyplot(plot)
        except pd.errors.ParserError:
            st.error("Invalid file format. Please upload a valid Excel file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


# Start the application
if __name__ == "__main__":
    app()