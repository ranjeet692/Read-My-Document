import streamlit as st

from model import get_result

st.header("Easy Analytics")
file = st.file_uploader("Upload a csv file", type=["csv"])

if file is not None:
    query = st.text_area("Ask your questions, for example - 'List top ten 4+ star rating restaurants in  format: restaurant name, rating and cuisine'")
    submit = st.button("Submit")
    if submit:
        st.write(get_result(file, query))