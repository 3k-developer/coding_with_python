'''
For smooth execution of this script, we have to use the below mentioned version of the following libraries:
fbprophet 0.7.1
pandas 1.0.4
pystan 2.19.1.1
numpy 1.19.1
python greater than 3.6 but less than 3.8.5

Note: It is highly recommended to use Anaconda environment for easy version navigation of libraries.
'''

import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# To Define Start and End date for our model
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# To Set Title for our app
st.title("Stock Prediction App")

# To list out the stocks for which we want to generate prediction
stocks = ("AAPL", "GOOG", "MSFT", "TSLA", "GME")

# To select between the stock options for prediction
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# To add slider for number of years for prediction
# we have selected 1 and 4 as start and end value for years
n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365  # To make days available to our prediction model

# To load the stock data from yahoo finance
# We are also using cache decorator of Streamlit to avoid multiple fetch of same data


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

# To display our raw data fetched
st.subheader("Raw data")
st.write(data.tail())

# To plot the data fetched


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Forecasting with Facebook Prophet
df_train = data[['Date', 'Close']]
# Entering data into model as accepted by it
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Creating facebook prophet model and prediction future
m = Prophet(daily_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# To display and plot the forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# To execute the script enter the following command in terminal
# streamlit run main.py
