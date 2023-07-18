# importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from bokeh.plotting import figure
import xlrd
import warnings
warnings.filterwarnings("ignore")

st.title('Oil Price Prediction')

#load dataset to plot alongside predictions
df = pd.read_excel("Crude_oil_WTI.xls")
df['Date'] = pd.to_datetime(df['Date'], format='%dd')
df.set_index(['Date'], inplace=True)
df.head()



    
# Add a date input widget
date_input = st.date_input("Enter a date:")
predict_button = st.button("Predict", key="predict_button")

    
if predict_button:
        input_date = pd.to_datetime(date_input)
        
        # Generate predictions for the input date and beyond
        model=pickle.load(open('lstm_model.pkl','rb'))
        future_dates = pd.date_range(start=df['Date'].min(), end=input_date, freq='D')
        predictions = model.predict(pd.DataFrame({'ds': future_dates}))
        
        # Get the forecasted price for the input date
        forecasted_price = predictions.loc[predictions['ds'] == input_date, 'yhat'].values
        if len(forecasted_price) > 0:
            forecasted_price = forecasted_price[0]
            st.write("Forecasted Oil Price (USD/BBL):", forecasted_price)
        else:
            st.write("No forecast available for the specified date.")

       

        # Visualize the graph
        st.subheader("Oil Price Prediction")
        p = figure(x_axis_type='datetime', title='Oil Price Prediction', width=800, height=400)
        p.line(df['Date'], df['Price'], line_color='blue', legend_label='Actual Price')
        p.line(predictions['ds'], predictions['yhat'], line_color='green', legend_label='Forecasted Price')
    

