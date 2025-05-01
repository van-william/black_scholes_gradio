from gradio_client import Client
import unittest
import time

class TestBlackScholesCalculator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test class - wait for server to be ready"""
        cls.client = Client("http://127.0.0.1:7861")
        # Give the server a moment to start up
        time.sleep(2)
    
    def test_aapl_call_option(self):
        """Test AAPL call option calculation"""
        print("\nTest Case 1: AAPL Call Option")
        try:
            result = self.client.predict(
                "AAPL",                  # ticker
                150.0,                   # strike_price
                30,                      # days_to_expiration
                0.05,                    # risk_free_rate
                0.005,                   # dividend_rate
                "call",                  # option_type
                api_name="/predict"
            )
            
            # Unpack the result (returns [result_text, plot])
            result_text, plot = result
            
            # Basic validation
            self.assertIsInstance(result_text, str)
            self.assertIn("Option Price: $", result_text)
            self.assertIn("Current Stock Price: $", result_text)
            self.assertIn("Implied Volatility:", result_text)
            
            print(result_text)
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")
    
    def test_tsla_put_option(self):
        """Test TSLA put option calculation"""
        print("\nTest Case 2: TSLA Put Option")
        try:
            result = self.client.predict(
                "TSLA",                  # ticker
                200.0,                   # strike_price
                60,                      # days_to_expiration
                0.05,                    # risk_free_rate
                0.0,                     # dividend_rate
                "put",                   # option_type
                api_name="/predict"
            )
            
            # Unpack the result (returns [result_text, plot])
            result_text, plot = result
            
            # Basic validation
            self.assertIsInstance(result_text, str)
            self.assertIn("Option Price: $", result_text)
            self.assertIn("Current Stock Price: $", result_text)
            self.assertIn("Implied Volatility:", result_text)
            
            print(result_text)
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")
    
    def test_invalid_ticker(self):
        """Test handling of invalid ticker"""
        print("\nTest Case 3: Invalid Ticker")
        try:
            result = self.client.predict(
                "INVALID",               # ticker
                100.0,                   # strike_price
                30,                      # days_to_expiration
                0.05,                    # risk_free_rate
                0.0,                     # dividend_rate
                "call",                  # option_type
                api_name="/predict"
            )
            
            # Unpack the result (returns [error_message, None])
            error_message, plot = result
            
            # Validate error handling
            self.assertIsInstance(error_message, str)
            self.assertIn("An error occurred:", error_message)
            self.assertIsNone(plot)
            
            print(error_message)
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

    def test_extreme_values(self):
        """Test handling of extreme values"""
        print("\nTest Case 4: Extreme Values")
        try:
            result = self.client.predict(
                "AAPL",                  # ticker
                2000.0,                  # very high strike price
                365,                     # maximum days
                0.1,                     # maximum risk-free rate
                0.2,                     # maximum dividend rate
                "call",                  # option_type
                api_name="/predict"
            )
            
            # Unpack the result
            result_text, plot = result
            
            # Basic validation
            self.assertIsInstance(result_text, str)
            self.assertIn("Option Price: $", result_text)
            
            print(result_text)
        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    # First ensure the app is running
    print("Make sure the Gradio app is running (python app.py) before running tests!")
    print("Waiting 2 seconds before starting tests...")
    time.sleep(2)
    
    # Run the tests
    unittest.main(argv=[''], verbosity=2) 