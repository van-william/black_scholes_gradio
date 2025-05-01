import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import gradio as gr


def get_stock_data(ticker: str, start_date: datetime, end_date: datetime):
    """
    Fetch historical stock data for a given ticker and date range.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
    
    Returns:
        Historical stock data including Open, High, Low, Close prices
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    return hist

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, q: float = 0, option_type: str = 'call') -> float:
    """
    Calculate option price using the Black-Scholes model.
    
    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (decimal)
        sigma (float): Volatility (decimal)
        q (float, optional): Dividend yield (decimal). Defaults to 0.
        option_type (str, optional): Type of option ('call' or 'put'). Defaults to 'call'.
    
    Returns:
        float: Option price
    
    Raises:
        ValueError: If option_type is not 'call' or 'put'
    """
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

def calculate_option_price(
    ticker: str,
    strike_price: float,
    days_to_expiration: int,
    risk_free_rate: float,
    dividend_rate: float,
    option_type: str
) -> tuple[float, float, float]:
    """
    Calculate option price and related metrics for a given stock.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        strike_price (float): Strike price of the option
        days_to_expiration (int): Number of days until option expiration (1-365)
        risk_free_rate (float): Risk-free interest rate (0.0-0.1)
        dividend_rate (float): Dividend yield (0.0-0.2)
        option_type (str): Type of option ('call' or 'put')
    
    Returns:
        tuple[float, float, float]: (option_price, current_stock_price, volatility)
    
    Raises:
        ValueError: If parameters are invalid or if unable to fetch stock data
    """
    # Input validation
    if not isinstance(ticker, str) or not ticker:
        raise ValueError("Ticker must be a non-empty string")
    if not 0 < strike_price:
        raise ValueError("Strike price must be positive")
    if not 1 <= days_to_expiration <= 365:
        raise ValueError("Days to expiration must be between 1 and 365")
    if not 0 <= risk_free_rate <= 0.1:
        raise ValueError("Risk-free rate must be between 0 and 0.1")
    if not 0 <= dividend_rate <= 0.2:
        raise ValueError("Dividend rate must be between 0 and 0.2")
    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be 'call' or 'put'")

    # Get current stock price
    stock = yf.Ticker(ticker)
    
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
    """
    Create an interactive candlestick chart for the given stock.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    
    Returns:
        go.Figure: Plotly figure object containing the candlestick chart
        
    Raises:
        ValueError: If unable to fetch stock data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    data = get_stock_data(ticker, start_date, end_date)

    if data.empty:
        raise ValueError(f"No historical data available for {ticker}")

    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

    fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price')
    return fig

def app_interface(
    ticker: str,
    strike_price: float,
    days_to_expiration: int,
    risk_free_rate: float,
    dividend_rate: float,
    option_type: str
) -> tuple[str, go.Figure]:
    """
    Main interface function for the Black-Scholes option calculator.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        strike_price (float): Strike price of the option
        days_to_expiration (int): Number of days until option expiration (1-365)
        risk_free_rate (float): Risk-free interest rate (0.0-0.1)
        dividend_rate (float): Dividend yield (0.0-0.2)
        option_type (str): Type of option ('call' or 'put')
    
    Returns:
        tuple[str, go.Figure]: (Formatted results string, Stock price chart)
    """
    try:
        option_price, current_price, volatility = calculate_option_price(
            ticker, strike_price, days_to_expiration, risk_free_rate, dividend_rate, option_type
        )
        stock_chart = plot_stock_data(ticker)
        
        result = f"""
        Option Price: ${option_price:.2f}
        Current Stock Price: ${current_price:.2f}
        Implied Volatility: {volatility:.2%}
        Strike Price: ${strike_price:.2f}
        Days to Expiration: {days_to_expiration}
        Risk-Free Rate: {risk_free_rate:.1%}
        Dividend Rate: {dividend_rate:.1%}
        Option Type: {option_type.capitalize()}
        """
        
        return result, stock_chart
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return error_message, None

# Create Gradio interface with improved documentation
iface = gr.Interface(
    fn=app_interface,
    inputs=[
        gr.Textbox(
            label="Stock Ticker",
            placeholder="Enter stock symbol (e.g., AAPL, TSLA)",
            info="The stock ticker symbol to calculate options for"
        ),
        gr.Number(
            label="Strike Price",
            info="The price at which the option can be exercised"
        ),
        gr.Slider(
            minimum=1,
            maximum=365,
            step=1,
            label="Days to Expiration",
            info="Number of days until the option expires (1-365)"
        ),
        gr.Slider(
            minimum=0,
            maximum=0.1,
            step=0.001,
            label="Risk-Free Rate",
            info="The risk-free interest rate (0-10%)"
        ),
        gr.Slider(
            minimum=0,
            maximum=0.2,
            step=0.001,
            label="Dividend Rate",
            info="Annual dividend yield (0-20%)"
        ),
        gr.Radio(
            choices=["call", "put"],
            label="Option Type",
            info="Call options give the right to buy, put options give the right to sell"
        )
    ],
    outputs=[
        gr.Textbox(label="Option Pricing Results"),
        gr.Plot(label="Stock Price Chart")
    ],
    title="Black-Scholes Option Pricing Calculator",
    description="""
    Calculate option prices using the Black-Scholes model and view stock price charts.
    
    This calculator helps you estimate the fair value of stock options based on:
    - Current stock price (fetched automatically)
    - Strike price (the price at which the option can be exercised)
    - Time to expiration (in days)
    - Risk-free interest rate
    - Stock's dividend yield
    - Historical volatility (calculated automatically)
    
    The calculator also provides a candlestick chart of the stock's recent price history.
    """,
    examples=[
        ["AAPL", 200, 30, 0.05, 0.005, "call"],
        ["TSLA", 300, 60, 0.05, 0.0, "put"],
        ["MSFT", 400, 90, 0.04, 0.01, "call"],
    ]
)

# Launch the app with MCP server
if __name__ == "__main__":
    iface.launch()