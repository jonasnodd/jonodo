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

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Your helper functions (natural_sort_key, contains_alpha) here

def app():
    st.title("Clustering")
    st.write("Here you can insert an excel-sheet with data and cluster the preferred data using DBSCAN method.")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
    if uploaded_file is not None:
        # Allow the user to select the sheet within the Excel file
        sheet_name = st.selectbox("Select the sheet", pd.ExcelFile(uploaded_file).sheet_names)

        # Allow the user to specify how many rows to skip (optional)
        skip_rows = st.number_input("Skip how many rows?", min_value=0, max_value=100, step=1, value=0)

        # Load the data from the selected sheet with optional row skipping
        dataframe = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skip_rows)

        # Allow the user to select columns that should not contain strings
        columns_no_strings = st.multiselect('Select columns without strings', dataframe.columns)

        # Remove rows with strings in the selected columns
        dataframe = dataframe.dropna(subset=columns_no_strings, how='any', axis=0)

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

if __name__ == "__main__":
    app()









