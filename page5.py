import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

def natural_sort_key(s):
    return tuple(int(part) if re.match(r'^\d+$', part) else part for part in re.split(r'(\d+)', s))

def identify_flat_parts_smoothed(time_series, window_length=5, polyorder=2, threshold=0.1):
    """
    Identify flat parts of a time series using a smoothed derivative.

    Parameters:
        time_series (numpy array or list): The input time series data.
        window_length (int): The length of the window used for the smoothing.
        polyorder (int): The order of the polynomial used in the smoothing function.
        threshold (float): The threshold to determine flatness. A smaller threshold
                           will result in more segments being considered flat.

    Returns:
        List of tuples: Each tuple represents a flat segment and contains the start and
                        end indices of the segment.
    """
    # Convert the input to a NumPy array for easier calculations
    time_series = np.array(time_series)

    # Smooth the time series using Savitzky-Golay filter
    smoothed_series = savgol_filter(time_series, window_length, polyorder)

    # Find the segments where the difference between the original and smoothed series is small (i.e., flat regions)
    flat_segments = []
    start = None
    for i in range(len(time_series)):
        if abs(time_series[i] - smoothed_series[i]) < threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i - 1
                if end-start >= 30:
                    flat_segments.append((start, end))
                start = None

    # Check if the last segment was flat and add it if necessary
    if start is not None:
        flat_segments.append((start, len(time_series) - 1))

    return flat_segments

def app():
    st.title("Page 5")
    st.write("This is the content of Page 5.")
    uploaded_files = st.file_uploader("Choose multiple files", accept_multiple_files=True)

    if uploaded_files:
        dfs = []  # List to store DataFrames from each file

        for uploaded_file in uploaded_files:
            # Step 1: Read the uploaded file and parse JSON content
            try:
                data = json.loads(uploaded_file.read())
            except Exception as e:
                st.error(f"Error reading JSON content from file {uploaded_file.name}: {e}")
                continue

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(data)

            # Convert the first column to datetime format and set the timezone to Oslo
            df["tid"] = pd.to_datetime(df["tid"]).dt.tz_convert('Europe/Oslo')

            # Step 2: Apply alphabetical numbering to non-numeric columns
            non_numeric_columns = df.select_dtypes(include='object').columns
            alphabetical_numbers = {value: index + 1 for index, value in enumerate(sorted(df[non_numeric_columns].stack().unique(), key=natural_sort_key))}
            df[non_numeric_columns] = df[non_numeric_columns].applymap(lambda x: alphabetical_numbers[x] if x in alphabetical_numbers else x)

            # Append the DataFrame to the list
            dfs.append(df)

        # Step 3: Concatenate all DataFrames into a single DataFrame
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)

            st.success("DataFrames created and combined successfully!")
            st.dataframe(combined_df)

            # Step 4: Select the x-axis column using a select box
            x_axis_column = st.selectbox("Select X-axis column", combined_df.columns)

            # Step 5: Plot the data over time
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()  # Create a secondary y-axis
            first_axis = st.multiselect('Primary y-axis', options=combined_df.columns)
            second_axis = st.multiselect('Secondary y-axis', options=combined_df.columns)
            for column in combined_df.columns:
                if column != x_axis_column:  # Skip the x-axis column
                    if column in second_axis:
                        # Plot "BT_A" and "BT_B" on the secondary y-axis
                        ax2.plot(combined_df[x_axis_column], combined_df[column], label=column, linestyle='dashed', alpha=0.7)
                    elif column in first_axis:
                        # Plot other columns on the primary y-axis
                        ax1.plot(combined_df[x_axis_column], combined_df[column], label=column)

            ax1.set_xlabel(x_axis_column)
            ax1.set_ylabel("Value (Primary Y-axis)")
            ax2.set_ylabel("Value (Secondary Y-axis)")
            ax1.set_title("Data Over Time")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig)

            df_notime = combined_df[combined_df.columns[~combined_df.columns.isin(['tid'])]]
            rolling_mean = df_notime.rolling(10,center=True).mean()

            flat_segments = identify_flat_parts_smoothed(rolling_mean[first_axis[0]], window_length=2, polyorder=1, threshold=2)
            st.write(flat_segments)  # Output: [(0, 6), (7, 9), (10, 12)]

            # Split the time series based on flat segments and plot the resulting subseries
            fig,ax = plt.subplots(figsize=(10, 6))
            for i in (0,len(flat_segments)-1):
                if i+1 == len(flat_segments):
                    ax.plot(combined_df[x_axis_column][flat_segments[i][1]:],combined_df[first_axis[0]][flat_segments[i][1]:])
                else:
                    ax.plot(combined_df[x_axis_column][flat_segments[i][1]:flat_segments[i+1][0]],combined_df[first_axis[0]][flat_segments[i][1]:flat_segments[i+1][0]])
                st.write(flat_segments[i][1])
            ax.set_xlabel(x_axis_column)
            ax.set_ylabel("Value")
            ax.set_title("Subseries Data Over Time")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    app()
