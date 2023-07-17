#Importing required dependencies
import streamlit as st
from datetime import date
import pandas as pd
import matplotlib as plt
import seaborn as sns
from plotly import graph_objs as go

START = "1986-01-02"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Crude Oil Price Prediction')

@st.cache
df = pd.read_excel('Crude_oil_WTI.xls')
 

st.subheader('Raw data')
st.write(df.head())


def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['Date'], y=df['Price']))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


