import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import pandas as pd
import statsmodels.api as sm

def natural_sort_key(s):
    return tuple(int(part) if re.match(r'^\d+$', part) else part for part in re.split(r'(\d+)', s))

def app():
    st.title("Page 2")
    st.write("This is the content of Page 2.")
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
            first_axis = st.multiselect('primary y-axis', options=combined_df.columns)
            second_axis = st.multiselect('secondary y-axis', options=combined_df.columns)
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

if __name__ == "__main__":
    app()