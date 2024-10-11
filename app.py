import gradio as gr
import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from datetime import datetime, timedelta


def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist

def black_scholes(S, K, T, r, sigma, q=0, option_type='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - 
                        K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == 'put':
        option_price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - 
                        S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    return option_price

def calculate_option_price(ticker, strike_price, days_to_expiration, risk_free_rate, dividend_rate, option_type):
    # Get current stock price
    stock = yf.Ticker(ticker)
    
    # Try to get the current price from different sources
    try:
        current_price = stock.info.get('currentPrice')
        if current_price is None:
            current_price = stock.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        raise ValueError(f"Unable to fetch current price for {ticker}. Error: {str(e)}")
    
    # Calculate historical volatility (last 252 trading days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=252)
    hist_data = get_stock_data(ticker, start_date, end_date)
    
    if hist_data.empty:
        raise ValueError(f"No historical data available for {ticker}")
    
    returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
    sigma = returns.std() * np.sqrt(252)
    
    # Calculate option price
    T = days_to_expiration / 365
    option_price = black_scholes(current_price, strike_price, T, risk_free_rate, sigma, dividend_rate, option_type)
    
    return option_price, current_price, sigma

def plot_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    data = get_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
    
    fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price')
    return fig

def app_interface(ticker, strike_price, days_to_expiration, risk_free_rate, dividend_rate, option_type):
    try:
        option_price, current_price, volatility = calculate_option_price(ticker, strike_price, days_to_expiration, risk_free_rate, dividend_rate, option_type)
        stock_chart = plot_stock_data(ticker)
        
        result = f"""
        Option Price: ${option_price:.2f}
        Current Stock Price: ${current_price:.2f}
        Implied Volatility: {volatility:.2%}
        """
        
        return result, stock_chart
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return error_message, None

# Create Gradio interface
iface = gr.Interface(
    fn=app_interface,
    inputs=[
        gr.Textbox(label="Stock Ticker"),
        gr.Number(label="Strike Price"),
        gr.Slider(minimum=1, maximum=365, step=1, label="Days to Expiration"),
        gr.Slider(minimum=0, maximum=0.1, step=0.001, label="Risk-Free Rate"),
        gr.Slider(minimum=0, maximum=0.2, step=0.001, label="Dividend Rate"),
        gr.Radio(["call", "put"], label="Option Type")
    ],
    outputs=[
        gr.Textbox(label="Option Pricing Results"),
        gr.Plot(label="Stock Price Chart")
    ],
    title="Black-Scholes Option Pricing Calculator",
    description="Calculate option prices using the Black-Scholes model and view stock price charts."
)

# Launch the app
iface.launch()