#Importing required dependencies
import streamlit as st
from datetime import date
import pandas as pd
import matplotlib as plt
import seaborn as sns
from plotly import graph_objs as go
import xlrd 

 

def main():
    # Set the app title
    st.title("Oil Price Prediction")

    # Apply custom CSS styling
    st.markdown(
        """
        <style>
            /* Change the font family and color of the heading */
            .title-wrapper {
                font-family: 'Arial', sans-serif;
                color: blue;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    

df = xlrd.open_workbook('Crude_oil_WTI.xls')

# Storing the first sheet into a variable
sheet = df.sheet_by_index(0)

# Get max no of rows and columns
print("Number of Rows: ", sheet.nrows)
print("Number of Columns: ",sheet.ncols)

# Get first 10 rows for 5 columns
for i in range(11):
    for j in range(2):
        print(sheet.cell_value(i,j), end="\t\t\t")
    print()
 


def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Price']))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

