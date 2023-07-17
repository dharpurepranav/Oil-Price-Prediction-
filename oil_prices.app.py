#Importing required dependencies
import streamlit as st
from datetime import date
import pandas as pd
import matplotlib as plt
import seaborn as sns
from plotly import graph_objs as go

 

def main():
    # Set the app title
    st.title("Prophet Model Deployment G-3")

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
    
	
df = pd.read_excel('Crude_oil_WTI.xls')
 

st.subheader('Raw data')
st.write(df.head())


def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Price']))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


