from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import RSI, SMA
from surmount.logging import log
import numpy as np

class TradingStrategy(Strategy):
    
    def __init__(self):
        self.assets_list = ["NVDA", "MSFT", "GOOGL", "SNOW", "PLTR", "ASML", "TSLA"]
        self.momentum_window_short = 63  # 3 months (trading days)
        self.momentum_window_long = 126  # 6 months (trading days)
        self.stop_loss_threshold = 0.12
        self.profit_take_threshold = 0.30
        self.rebalance_frequency = "1month"
        self.last_peak = {}

    @property
    def interval(self):
        return "1day"

    @property
    def assets(self):
        return self.assets_list

    def calculate_momentum_score(self, data, ticker):
        """ Calculate momentum score: (3-month return + 6-month return) / volatility """
        try:
            prices = [candle["close"] for candle in data[ticker][-self.momentum_window_long:]]
            if len(prices) < self.momentum_window_long:
                return 0

            returns_short = prices[-1] / prices[-self.momentum_window_short] - 1
            returns_long = prices[-1] / prices[0] - 1
            vol = np.std(np.diff(np.log(prices)))

            return (returns_short + returns_long) / (vol + 1e-6)  # Avoid division by zero
        except Exception as e:
            log(f"Momentum calc error for {ticker}: {e}")
            return 0

    def run(self, data):
        allocation = {}
        momentum_scores = {}

        for ticker in self.assets_list:
            if ticker not in data["ohlcv"]:
                continue

            # Compute momentum score
            momentum_scores[ticker] = self.calculate_momentum_score(data["ohlcv"], ticker)
        
        # Normalize momentum scores to determine weighting
        total_score = sum(momentum_scores.values()) or 1  # Avoid division by zero
        for ticker in self.assets_list:
            allocation[ticker] = max(momentum_scores[ticker] / total_score, 0)

        # Apply risk-based weighting: lower volatility assets get more weight
        volatilities = {ticker: np.std(np.diff([candle["close"] for candle in data["ohlcv"].get(ticker, [])])) for ticker in self.assets_list}
        total_volatility = sum(1/(vol + 1e-6) for vol in volatilities.values())

        for ticker in self.assets_list:
            allocation[ticker] *= (1/(volatilities[ticker] + 1e-6)) / total_volatility

        # Stop-loss & Profit-taking logic
        for ticker in self.assets_list:
            price_data = data["ohlcv"].get(ticker, [])
            if len(price_data) < 2:
                continue

            current_price = price_data[-1]["close"]
            prev_price = price_data[-2]["close"]

            # Track highest price for stop-loss
            self.last_peak[ticker] = max(self.last_peak.get(ticker, current_price), current_price)

            # Stop-loss: if price drops 12% from peak
            if current_price < self.last_peak[ticker] * (1 - self.stop_loss_threshold):
                allocation[ticker] *= 0.5  # Reduce exposure

            # Profit-taking: if price jumps >30% in a month
            if prev_price > 0 and (current_price / prev_price - 1) > self.profit_take_threshold:
                allocation[ticker] *= 0.8  # Trim position

            # RSI-based trimming
            try:
                rsi = RSI(ticker, data["ohlcv"], 14)[-1]
                if rsi > 80:
                    allocation[ticker] *= 0.85  # Reduce allocation if overbought
            except Exception as e:
                log(f"RSI calculation error for {ticker}: {e}")

        # Normalize allocations to sum to 1
        total_alloc = sum(allocation.values()) or 1
        for ticker in self.assets_list:
            allocation[ticker] /= total_alloc

        return TargetAllocation(allocation)