from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import RSI, SMA, STDEV
from surmount.logging import log
import pandas as pd
import numpy as np


class TradingStrategy(Strategy):
   def __init__(self):
       # Define the assets to trade
       self.tickers = ["NVDA", "MSFT", "GOOGL", "SNOW", "PLTR", "ASML", "TSLA"]
       self.data_list = []  # No additional data sources needed for this strategy


   @property
   def interval(self):
       return "1day"  # Daily data for monthly rebalancing


   @property
   def assets(self):
       return self.tickers


   @property
   def data(self):
       return self.data_list


   def run(self, data):
       # Access OHLCV data
       ohlcv = data["ohlcv"]
       if len(ohlcv) < 200:  # Ensure sufficient data for 200-day MA
           log("Insufficient data for strategy execution")
           return TargetAllocation({ticker: 0 for ticker in self.tickers})


       allocation_dict = {}
       momentum_scores = {}
       volatilities = {}


       # Calculate momentum score and volatility for each asset
       for ticker in self.tickers:
           closes = [entry[ticker]["close"] for entry in ohlcv]
           if len(closes) < 200:
               momentum_scores[ticker] = 0
               volatilities[ticker] = 1
               continue


           # Momentum Score: (3-month + 6-month return) / volatility
           three_month_return = (closes[-1] / closes[-63] - 1) if len(closes) >= 63 else 0  # Approx 3 months
           six_month_return = (closes[-1] / closes[-126] - 1) if len(closes) >= 126 else 0  # Approx 6 months
           volatility = STDEV(ticker, ohlcv, 20)[-1] / closes[-1] if STDEV(ticker, ohlcv, 20) else 1  # 20-day std dev
           momentum_scores[ticker] = (three_month_return + six_month_return) / max(volatility, 0.01)  # Avoid division by zero
           volatilities[ticker] = volatility


           # Apply profit-taking rule
           one_month_return = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0  # Approx 1 month
           rsi = RSI(ticker, ohlcv, 14)[-1] if RSI(ticker, ohlcv, 14) else 50
           if one_month_return > 0.3 or rsi > 80:
               log(f"Profit-taking triggered for {ticker}: Return={one_month_return:.2%}, RSI={rsi:.2f}")
               momentum_scores[ticker] *= 0.85  # Reduce exposure by 15% (trim position)


           # Apply stop-loss rule
           peak_price = max(closes[-63:])  # Last 3 months peak
           drop_from_peak = (peak_price - closes[-1]) / peak_price
           sma_50 = SMA(ticker, ohlcv, 50)[-1] if SMA(ticker, ohlcv, 50) else closes[-1]
           sma_200 = SMA(ticker, ohlcv, 200)[-1] if SMA(ticker, ohlcv, 200) else closes[-1]
           if drop_from_peak > 0.12 or sma_50 < sma_200:
               log(f"Stop-loss triggered for {ticker}: Drop={drop_from_peak:.2%}, SMA50={sma_50:.2f}, SMA200={sma_200:.2f}")
               momentum_scores[ticker] = 0  # Temporarily remove stock


       # Adjust exposure based on momentum score
       for ticker in self.tickers:
           ms = momentum_scores[ticker]
           if ms < 0:
               momentum_scores[ticker] *= 0.5  # Reduce exposure by 50% if negative momentum


       # Risk-based weighting: Wi = 1 / volatility
       total_inverse_vol = sum(1 / max(volatilities[ticker], 0.01) for ticker in self.tickers if momentum_scores[ticker] > 0)
       if total_inverse_vol == 0:
           log("No valid allocations; all momentum scores are zero or negative")
           return TargetAllocation({ticker: 0 for ticker in self.tickers})


       # Calculate initial weights based on momentum and volatility
       for ticker in self.tickers:
           if momentum_scores[ticker] > 0:
               weight = (1 / max(volatilities[ticker], 0.01)) / total_inverse_vol
               allocation_dict[ticker] = weight * min(1, momentum_scores[ticker] / max(momentum_scores.values()))  # Scale by momentum
           else:
               allocation_dict[ticker] = 0


       # Normalize allocations to sum between 0 and 1
       total_allocation = sum(allocation_dict.values())
       if total_allocation > 0:
           for ticker in self.tickers:
               allocation_dict[ticker] /= total_allocation
               allocation_dict[ticker] = min(max(allocation_dict[ticker], 0), 1)  # Ensure bounds
               log(f"{ticker}: Momentum Score={momentum_scores[ticker]:.2f}, Volatility={volatilities[ticker]:.2f}, Weight={allocation_dict[ticker]:.2%}")
       else:
           allocation_dict = {ticker: 0 for ticker in self.tickers}


       return TargetAllocation(allocation_dict)