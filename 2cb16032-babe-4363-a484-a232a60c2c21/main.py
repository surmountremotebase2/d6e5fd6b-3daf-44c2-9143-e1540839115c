from surmount.base_class import Strategy, TargetAllocation
from surmount.technical_indicators import RSI, SMA
from surmount.logging import log
import numpy as np
from datetime import datetime

class TradingStrategy(Strategy):
    def __init__(self):
        # Tech growth stocks
        self.tickers = ["NVDA", "MSFT", "GOOGL", "SNOW", "PLTR", "ASML", "TSLA"]
        
        # Strategy parameters
        self.momentum_lookback_3m = 63   # ~3 months of trading days
        self.momentum_lookback_6m = 126  # ~6 months of trading days
        self.volatility_lookback = 30    # For calculating standard deviation
        self.rsi_period = 14             # For RSI calculation
        self.profit_taking_threshold = 0.30  # 30% gain for profit taking
        self.profit_taking_amount = 0.15     # Take 15% profit
        self.stop_loss_threshold = 0.12      # 12% drawdown for stop loss
        self.ma_short = 50                   # 50-day MA
        self.ma_long = 200                   # 200-day MA
        self.rsi_overbought = 80             # RSI overbought threshold
        
        # Tracking variables
        self.last_rebalance_date = None
        self.peak_prices = {}  # To track ATHs for stop loss
        self.last_allocation = {}  # To track previous allocation
        self.previous_weights = {}  # For smooth transitions

    @property
    def interval(self):
        return "1day"
    
    @property
    def assets(self):
        return self.tickers
    
    def calculate_momentum_score(self, ticker, data):
        """Calculate momentum score based on 3-month and 6-month returns divided by volatility"""
        closes = [bar[ticker]["close"] for bar in data if ticker in bar]
        
        if len(closes) < self.momentum_lookback_6m + 1:
            log(f"Not enough data for {ticker} momentum calculation")
            return 0
        
        # Calculate returns
        current_price = closes[-1]
        price_3m_ago = closes[-min(self.momentum_lookback_3m, len(closes))]
        price_6m_ago = closes[-min(self.momentum_lookback_6m, len(closes))]
        
        return_3m = (current_price / price_3m_ago) - 1
        return_6m = (current_price / price_6m_ago) - 1
        
        # Calculate volatility (standard deviation of daily returns)
        recent_prices = closes[-min(self.volatility_lookback, len(closes)):]
        daily_returns = [recent_prices[i]/recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
        
        if not daily_returns:
            return 0
            
        volatility = np.std(daily_returns)
        
        # Avoid division by zero
        if volatility == 0:
            volatility = 0.001
            
        # Calculate momentum score
        momentum_score = (return_3m + return_6m) / volatility
        
        return momentum_score
    
    def check_death_cross(self, ticker, data):
        """Check if 50-day MA is below 200-day MA (death cross)"""
        try:
            ma_short_values = SMA(ticker, data, self.ma_short)
            ma_long_values = SMA(ticker, data, self.ma_long)
            
            if len(ma_short_values) < 2 or len(ma_long_values) < 2:
                return False
                
            # Current cross
            current_death_cross = ma_short_values[-1] < ma_long_values[-1]
            
            return current_death_cross
        except Exception as e:
            log(f"Error calculating death cross for {ticker}: {str(e)}")
            return False
    
    def check_stop_loss(self, ticker, current_price):
        """Check if price has fallen more than threshold % from peak"""
        if ticker not in self.peak_prices:
            self.peak_prices[ticker] = current_price
            return False
            
        # Update peak price if current price is higher
        if current_price > self.peak_prices[ticker]:
            self.peak_prices[ticker] = current_price
            
        drawdown = 1 - (current_price / self.peak_prices[ticker])
        
        return drawdown > self.stop_loss_threshold
    
    def check_monthly_rebalance(self, current_date):
        """Check if it's time for monthly rebalance"""
        if self.last_rebalance_date is None:
            self.last_rebalance_date = current_date
            return True
            
        # Parse dates
        current = datetime.strptime(current_date, "%Y-%m-%d")
        previous = datetime.strptime(self.last_rebalance_date, "%Y-%m-%d")
        
        # Check if month has changed
        if current.month != previous.month or current.year != previous.year:
            self.last_rebalance_date = current_date
            return True
            
        return False
        
    def run(self, data):
        ohlcv_data = data["ohlcv"]
        holdings = data["holdings"]
        
        # Need enough data for calculations
        if len(ohlcv_data) < self.momentum_lookback_6m:
            log(f"Not enough historical data yet. Have {len(ohlcv_data)} days, need {self.momentum_lookback_6m}")
            return None
            
        current_date = ohlcv_data[-1][self.tickers[0]]["date"].split(" ")[0]  # Extract date part
        
        # Check if monthly rebalance is needed
        monthly_rebalance = self.check_monthly_rebalance(current_date)
        
        # Skip if not monthly rebalance and no significant holdings changes
        if not monthly_rebalance and all(ticker in holdings for ticker in self.tickers):
            # Check for profit-taking or stop-loss conditions that would trigger mid-month
            trigger_rebalance = False
            
            for ticker in self.tickers:
                if ticker not in ohlcv_data[-1]:
                    continue
                    
                current_price = ohlcv_data[-1][ticker]["close"]
                
                # Get 30-day price change for profit taking
                if len(ohlcv_data) > 30 and ticker in ohlcv_data[-30]:
                    month_ago_price = ohlcv_data[-30][ticker]["close"]
                    monthly_return = current_price / month_ago_price - 1
                    
                    # Check profit taking for TSLA and NVDA
                    if ticker in ["TSLA", "NVDA"] and monthly_return > self.profit_taking_threshold:
                        log(f"Profit taking triggered for {ticker}: {monthly_return:.2%} return")
                        trigger_rebalance = True
                
                # Check RSI for overbought
                try:
                    rsi_value = RSI(ticker, ohlcv_data, self.rsi_period)[-1]
                    if rsi_value > self.rsi_overbought:
                        log(f"Overbought RSI triggered for {ticker}: {rsi_value:.2f}")
                        trigger_rebalance = True
                except Exception as e:
                    log(f"Error calculating RSI for {ticker}: {str(e)}")
                
                # Check stop loss
                if self.check_stop_loss(ticker, current_price):
                    log(f"Stop loss triggered for {ticker}")
                    trigger_rebalance = True
            
            if not trigger_rebalance:
                return None
        
        log(f"Running full rebalance on {current_date}")
        
        # Calculate momentum scores and volatilities
        momentum_scores = {}
        volatilities = {}
        
        for ticker in self.tickers:
            # Skip if ticker data not available
            if ticker not in ohlcv_data[-1]:
                momentum_scores[ticker] = 0
                volatilities[ticker] = float('inf')
                continue
                
            # Calculate momentum score
            momentum_scores[ticker] = float(self.calculate_momentum_score(ticker, ohlcv_data))
            
            # Calculate volatility for weighting
            closes = [bar[ticker]["close"] for bar in ohlcv_data[-self.volatility_lookback:] if ticker in bar]
            daily_returns = [closes[i]/closes[i-1] - 1 for i in range(1, len(closes))]
            volatilities[ticker] = float(np.std(daily_returns) if daily_returns else 1.0)
            
            # Avoid division by zero
            if volatilities[ticker] == 0:
                volatilities[ticker] = 0.001
                
            log(f"{ticker} - Momentum: {momentum_scores[ticker]:.4f}, Volatility: {volatilities[ticker]:.4f}")
            
        # Calculate raw weights based on inverse volatility and momentum
        raw_weights = {}
        total_positive_momentum = 0
        
        for ticker in self.tickers:
            # Apply death cross check
            death_cross = self.check_death_cross(ticker, ohlcv_data)
            if death_cross:
                log(f"Death cross detected for {ticker}, removing from allocation")
                raw_weights[ticker] = 0
                continue
                
            # Apply momentum score adjustment
            if momentum_scores[ticker] < 0:
                # Reduce exposure for negative momentum
                reduction = float(min(0.5, abs(momentum_scores[ticker]) * 0.5))  # Scale reduction, max 50%
                raw_weights[ticker] = float((1 / volatilities[ticker]) * (1 - reduction))
                log(f"{ticker} - Negative momentum, reducing weight by {reduction:.2%}")
            else:
                raw_weights[ticker] = float((1 / volatilities[ticker]) * (1 + momentum_scores[ticker]))
                total_positive_momentum += momentum_scores[ticker]
                
        # Profit taking logic for TSLA and NVDA
        for ticker in ["TSLA", "NVDA"]:
            if ticker not in ohlcv_data[-1] or ticker not in ohlcv_data[-30]:
                continue
                
            current_price = ohlcv_data[-1][ticker]["close"]
            month_ago_price = ohlcv_data[-30][ticker]["close"]
            monthly_return = current_price / month_ago_price - 1
            
            if monthly_return > self.profit_taking_threshold:
                log(f"Taking {self.profit_taking_amount:.2%} profit from {ticker} after {monthly_return:.2%} gain")
                raw_weights[ticker] *= (1 - self.profit_taking_amount)
        
        # Check RSI for trimming positions
        for ticker in self.tickers:
            if ticker not in ohlcv_data[-1]:
                continue
                
            try:
                rsi_value = RSI(ticker, ohlcv_data, self.rsi_period)[-1]
                if rsi_value > self.rsi_overbought:
                    trim_amount = min(0.3, (rsi_value - self.rsi_overbought) / 20)  # Scale trim amount
                    log(f"Trimming {ticker} by {trim_amount:.2%} due to high RSI: {rsi_value:.2f}")
                    raw_weights[ticker] *= (1 - trim_amount)
            except Exception as e:
                log(f"Error in RSI calculation for {ticker}: {str(e)}")
        
        # Normalize weights to sum to 1
        total_weight = sum(raw_weights.values())
        normalized_weights = {}
        
        if total_weight > 0:
            for ticker in self.tickers:
                normalized_weights[ticker] = float(raw_weights[ticker] / total_weight)
        else:
            # Equal weight fallback if all weights are zero
            for ticker in self.tickers:
                normalized_weights[ticker] = float(1.0 / len(self.tickers))
        
        # Smooth transition from previous weights (if available)
        final_weights = {}
        for ticker in self.tickers:
            if ticker in self.previous_weights:
                # Blend new and old weights (70% new, 30% old)
                final_weights[ticker] = float(0.7 * normalized_weights[ticker] + 0.3 * self.previous_weights.get(ticker, 0))
            else:
                final_weights[ticker] = float(normalized_weights[ticker])
        
        # Store for next run
        self.previous_weights = final_weights.copy()
        
        # Log final allocations
        log("Final allocations:")
        for ticker in self.tickers:
            log(f"{ticker}: {final_weights[ticker]:.2%}")
            
        # Ensure all values are float type for TargetAllocation
        target_allocation = {ticker: float(weight) for ticker, weight in final_weights.items()}
        
        return TargetAllocation(target_allocation)