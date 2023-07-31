import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def natural_sort_key(s):
    # Convert alphanumeric strings into tuples of integers and strings
    return tuple(int(part) if re.match(r'^\d+$', part) else part for part in re.split(r'(\d+)', s))

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

def app():
    st.title("Clustering")
    st.write("Here you can insert an excel-sheet with data and cluster the preferred data using DBSCAN method.")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
    if uploaded_file:
        try:
            # User Input: Options for data import
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            sheet_name = st.selectbox("Select a sheet", sheet_names, index=0)
            skip_rows_headers = st.number_input("Number of rows to skip above the column headers", min_value=0, value=0)
            skip_rows_data = st.number_input("Number of rows to skip after the column headers", min_value=0, value=0)
            cutoff_row_number = st.number_input("Row number to start cutting off data from", min_value=0, value=0)

            # Step 1: Read the selected Excel sheet (with user-specified options)
            df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=skip_rows_headers)

            # Step 2: Skip additional rows after column headers, if specified
            df = df.iloc[skip_rows_data:]

            # Step 3: Predict data types based on content and set the initial selections in multiselect
            string_columns = df.select_dtypes(include='object').columns
            date_columns = pd.to_datetime(df.select_dtypes(include='datetime').stack(), errors='coerce').unstack().columns
            numeric_columns = df.select_dtypes(include=['int', 'float']).columns

            # Step 4: Allow user to select column data types in the sidebar
            st.sidebar.write("**Column Data Types**")
            selected_string_columns = st.sidebar.multiselect("Select String columns", options=df.columns, default=list(string_columns))
            selected_date_columns = st.sidebar.multiselect("Select Date columns", options=df.columns, default=list(date_columns))

            # Step 5: Find the columns not selected as "String" or "Date" and assume them as "Numeric"
            selected_numeric_columns = list(set(df.columns) - set(selected_string_columns) - set(selected_date_columns))

            # Step 6: Apply data type conversions to the DataFrame
            for col in df.columns:
                if col in selected_string_columns:
                    df[col] = df[col].astype(str)
                elif col in selected_date_columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Step 7: Exclude rows from the cutoff row number if specified
            if cutoff_row_number > 0:
                df = df.iloc[:cutoff_row_number]

            # ... (Rest of the data processing steps, as required)

            st.success(f"DataFrame created successfully from '{uploaded_file.name}'!")
            st.dataframe(df)
            dataframe = df

            labels = np.array(dataframe.columns)

            options = st.multiselect('What do you want to trend?', labels)
            if len(options) >= 1:
                dataframe_integerized = dataframe[options].copy()
                for option in options:
                    dataframe_integerized[option] = contains_alpha(dataframe[option])

                sorted_columns = sorted(dataframe_integerized.columns, key=natural_sort_key)
                dataframe_integerized = dataframe_integerized[sorted_columns]

                features = dataframe_integerized.fillna(0).values
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)

                epsilon = st.slider('Epsilon', min_value=0.10, max_value=2.00, value=1.00, step=0.01)
                focus = st.selectbox('Which average should be presented?', options, index=0)
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

                st.pyplot(fig)

                pearsoncorr = pd.DataFrame(dataframe_integerized[options].corr(method='pearson'))

                plot, ax = plt.subplots()
                sns.heatmap(pearsoncorr, cmap='viridis', annot=True, fmt=".1f", linewidths=0.5, ax=ax)
                ax.set_title('Pearson Correlation Factor')

                # Display the plot using Streamlit
                st.pyplot(plot)
        except Exception as e:
            st.error(f"Error reading Excel file {uploaded_file.name}: {e}")

if __name__ == "__main__":
    app()









