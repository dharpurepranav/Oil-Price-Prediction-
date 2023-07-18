# importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prediction import predict
from PIL import Image
import xlrd
import warnings
warnings.filterwarnings("ignore")

 

#load dataset to plot alongside predictions
df = pd.read_excel("Crude_oil_WTI.xls")
df['Date'] = pd.to_datetime(df['Date'], format='%dd')
df.set_index(['Date'], inplace=True)


#page configuration
st.set_page_config(layout='centered')
 

year = st.slider("Select number of Years",1,30,step = 1)
    
    
pred = model.forecast(Date)
pred = pd.DataFrame(pred, columns=['Price'])
   
if st.button("Predict"):

        col1, col2 = st.columns([2,3])
        with col1:
             st.dataframe(pred)
        with col2:
            fig, ax = plt.subplots()
            df['Price'].plot(style='--', color='gray', legend=True, label='Actual')
            pred['Price'].plot(color='b', legend=True, label='prediction')
            st.pyplot(fig)
    