import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():
    st.title("Page 2")
    st.write("This is the content of Page 2.")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

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

            # Step 8: Select the x-axis column using a select box
            x_axis_column = st.selectbox("Select X-axis column", df.columns)

            # Step 9: Select the plot type (Line Plot or Scatter Plot)
            plot_type = st.radio("Select Plot Type", ["Line Plot", "Scatter Plot"], index=0)

            # Step 10: Plot the data for the sheet
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()  # Create a secondary y-axis
            first_axis = st.multiselect('primary y-axis', options=df.columns)
            second_axis = st.multiselect('secondary y-axis', options=df.columns)
            for column in df.columns:
                if column != x_axis_column:  # Skip the x-axis column
                    if column in second_axis:
                        # Plot on the secondary y-axis
                        if plot_type == "Scatter Plot":
                            ax2.scatter(df[x_axis_column], df[column], label=column, alpha=0.7)
                        else:
                            ax2.plot(df[x_axis_column], df[column], label=column, linestyle='dashed', alpha=0.7)
                    elif column in first_axis:
                        # Plot on the primary y-axis
                        if plot_type == "Scatter Plot":
                            ax1.scatter(df[x_axis_column], df[column], label=column, alpha=0.7)
                        else:
                            ax1.plot(df[x_axis_column], df[column], label=column)

            ax1.set_xlabel(x_axis_column)
            ax1.set_ylabel("Value (Primary Y-axis)")
            ax2.set_ylabel("Value (Secondary Y-axis)")
            ax1.set_title(f"{plot_type} from '{uploaded_file.name}'")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error reading Excel file {uploaded_file.name}: {e}")


if __name__ == "__main__":
    app()
