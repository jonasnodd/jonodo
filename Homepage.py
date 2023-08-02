import streamlit as st
from page1 import app as page1
from page2 import app as page2


st.set_page_config("Data Analysis Program")
st.title("Data Analysis Program")
st.write('This program is meant to make it possible for employees at GE Healthcare Lindesnes to analyze data from the production to find possible connections, deviations and improvements.')
st.write('As of 05.08.2023 there are still plenty of possible implementations to be made, but the Clustering module and the Batch Analysis module gives a slight idea of the possibilities of such a program.')
st.write('**Notes:**')
st.write('The Clustering module currently only takes 1 file on .xlsx format, while the Batch Analysis module can take multiple files of either .txt or .xlsx format (not both at once). Example format for .txt files and xlsx for Batch Analysis module attached in program folder under the "example"-folder. Be aware that the code has not been debugged thoroughly, and some errors can occur.')

# Define page titles and corresponding functions
PAGES = {
    "Page 1 - Clustering": page1,
    "Page 2 - Batch Analysis": page2,
}

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Run the selected page function
    page = PAGES[selection]
    if page:
        page()

if __name__ == "__main__":
    main()
