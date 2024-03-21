import streamlit as st
import numpy as np
import pandas as pd


st.set_page_config(layout="wide")

st.sidebar.markdown('## Car License Plate Numbers Database')
st.markdown('## Database')

csv_file = 'License_Plate_Database.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

st.dataframe(df)    