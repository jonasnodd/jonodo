import streamlit as st
from page1 import app as page1
from page2 import app as page2
from page3 import app as page3
from page4 import app as page4
from page5 import app as page5

st.title("Homepage")

# Define page titles and corresponding functions
PAGES = {
    "Home": None,
    "Page 1 - Clustering": page1,
    "Page 2": page2,
    "Page 3": page3,
    "Page 4 - Batch Analysis": page4,
    "Page 5": page5
}

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    if selection != "Home":
        st.sidebar.markdown('---')
        st.sidebar.markdown("[Home](.)")

    # Run the selected page function
    page = PAGES[selection]
    if page:
        page()

if __name__ == "__main__":
    main()
