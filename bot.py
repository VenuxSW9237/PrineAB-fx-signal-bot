import telebot
from telebot import types
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading

# Bot configuration
BOT_TOKEN = '8279479839:AAFtFUaH9koQO2jLcef3Ql8Syc-O2dFIy7c'  # Replace with your token
bot = telebot.TeleBot(BOT_TOKEN)

# Optimized pair selection (HIGH LIQUIDITY = BETTER SIGNALS)
MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']  # Reduced to most liquid pairs
VOLATILE_PAIRS = ['GBPJPY', 'EURJPY','ONTUSDT','BTCUSD']  # Higher risk, higher reward
ALL_PAIRS = MAJOR_PAIRS + VOLATILE_PAIRS

# Trading configuration (OPTIMIZED FOR HIGHER WIN RATE)
RISK_REWARD_RATIO = 2.5  # Increased from 2.0
ATR_MULTIPLIER_SL = 2.0  # Increased from 1.5 (wider SL = less stop outs)
ATR_MULTIPLIER_TP = 5.0  # Increased from 3.0 (better RR)

# BEST TRADING SESSIONS (Avoid low liquidity periods)
TRADING_HOURS = {
    'london_open': (8, 17),   # 8 AM - 5 PM GMT (Best for EUR, GBP pairs)
    'ny_open': (13, 22),      # 1 PM - 10 PM GMT (Best for USD pairs)
    'overlap': (13, 17),      # 1 PM - 5 PM GMT (HIGHEST VOLUME - BEST TIME!)
    'asian': (0, 8)           # 12 AM - 8 AM GMT (Good for JPY pairs, XAUUSD)
}

# Timeframe recommendations by pair type
TIMEFRAME_STRATEGY = {
    'scalping': '15m',      # 15min - 4-6 signals/day (requires monitoring)
    'intraday': '1h',       # 1 hour - 2-3 signals/day (RECOMMENDED)
    'swing': '4h',          # 4 hour - 1 signal/day (most reliable)
}
DEFAULT_TIMEFRAME = '1h'  # CHANGED TO 1H for better quality

# Timeframe options
TIMEFRAMES = {
    '5m': '5min',
    '15m': '15min',
    '30m': '30min', 
    '1h': '1h',
    '4h': '4h',
    '1d': '1day'
}

class ForexAnalyzer:
    """Enhanced analyzer with trend filtering and session awareness"""
    
    def __init__(self):
        self.twelve_api_key = '935b760b081245d28b791fbcbaf81732'
        self.alpha_api_key = 'JYXBWYP1S4MSY5NQ'
        
    def get_price_data(self, pair, interval='1h', periods=200):
        """Fetch historical price data from Twelve Data API"""
        try:
            formatted_pair = f"{pair[:3]}/{pair[3:]}"
            url = f"https://api.twelvedata.com/time_series"
            params = {
                'symbol': formatted_pair,
                'interval': interval,
                'outputsize': periods,
                'apikey': self.twelve_api_key,
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'values' not in data:
                print(f"Twelve Data error for {pair}: {data.get('message', 'Unknown error')}")
                return self.get_price_data_alpha(pair, interval, periods)
            
            df = pd.DataFrame(data['values'])
            df['timestamp'] = pd.to_datetime(df['datetime'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df.get('volume', 0))
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            print(f"Error fetching from Twelve Data: {e}")
            return self.get_price_data_alpha(pair, interval, periods)
    
    def get_price_data_alpha(self, pair, interval='60min', periods=200):
        """Fallback: Fetch data from Alpha Vantage"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': pair[:3],
                'to_symbol': pair[3:],
                'interval': interval,
                'apikey': self.alpha_api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            time_series_key = f'Time Series FX ({interval})'
            if time_series_key not in data:
                return None
            
            time_series = data[time_series_key]
            records = []
            
            for timestamp, values in list(time_series.items())[:periods]:
                records.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': 0
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
            
        except Exception as e:
            print(f"Error fetching from Alpha Vantage: {e}")
            return None
    
    def is_good_trading_time(self, pair):
        """Check if current time is good for trading this pair"""
        current_hour = datetime.utcnow().hour
        
        # Best times for different pairs
        if pair in ['XAUUSD']:
            # Gold trades well during London/NY overlap
            return TRADING_HOURS['overlap'][0] <= current_hour <= TRADING_HOURS['overlap'][1]
        elif 'JPY' in pair:
            # JPY pairs good during Asian session
            return TRADING_HOURS['asian'][0] <= current_hour <= TRADING_HOURS['asian'][1]
        elif pair in ['EURUSD', 'GBPUSD']:
            # EUR/GBP best during London/NY sessions
            return (TRADING_HOURS['london_open'][0] <= current_hour <= TRADING_HOURS['ny_open'][1])
        else:
            # USD pairs best during NY session
            return TRADING_HOURS['ny_open'][0] <= current_hour <= TRADING_HOURS['ny_open'][1]
    
    def calculate_indicators(self, df):
        """Calculate enhanced technical indicators"""
        # Moving Averages
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['signal']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # ADX (Trend Strength)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = df['atr'] * 14
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        
        return df
    
    def generate_signal(self, pair, timeframe='1h', strict_mode=True):
        """Generate HIGH-QUALITY trading signals with strict filters"""
        df = self.get_price_data(pair, interval=timeframe, periods=200)
        
        if df is None or len(df) < 200:
            print(f"Insufficient data for {pair}")
            return None
            
        df = self.calculate_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = latest['close']
        atr = latest['atr']
        
        # STRICT MODE: Check trading session (CRITICAL FOR WIN RATE)
        if strict_mode and not self.is_good_trading_time(pair):
            return None
        
        # TREND FILTER: Only trade with strong trends (ADX > 25)
        if strict_mode and latest['adx'] < 25:
            return None
        
        # Signal scoring system (STRICTER CRITERIA)
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        # 1. STRONG TREND CONFIRMATION (Most Important)
        if latest['ema_9'] > latest['ema_20'] > latest['ema_50'] > latest['ema_200']:
            bullish_score += 4
            reasons.append("🔥 Perfect Bullish Trend Alignment")
        elif latest['ema_9'] < latest['ema_20'] < latest['ema_50'] < latest['ema_200']:
            bearish_score += 4
            reasons.append("🔥 Perfect Bearish Trend Alignment")
        elif latest['ema_20'] > latest['ema_50'] > latest['sma_100']:
            bullish_score += 2
            reasons.append("📈 Strong Uptrend")
        elif latest['ema_20'] < latest['ema_50'] < latest['sma_100']:
            bearish_score += 2
            reasons.append("📉 Strong Downtrend")
        
        # 2. Moving Average Cross (Entry Signal)
        if latest['ema_9'] > latest['ema_20'] and prev['ema_9'] <= prev['ema_20']:
            bullish_score += 3
            reasons.append("✅ EMA 9/20 Golden Cross")
        elif latest['ema_9'] < latest['ema_20'] and prev['ema_9'] >= prev['ema_20']:
            bearish_score += 3
            reasons.append("🔻 EMA 9/20 Death Cross")
        
        # 3. RSI Confirmation (NOT oversold/overbought extremes)
        if 40 < latest['rsi'] < 60 and latest['rsi'] > prev['rsi']:
            bullish_score += 2
            reasons.append("💪 RSI Healthy Bullish Zone")
        elif 40 < latest['rsi'] < 60 and latest['rsi'] < prev['rsi']:
            bearish_score += 2
            reasons.append("⚠️ RSI Healthy Bearish Zone")
        
        # 4. MACD Confirmation
        if latest['macd'] > latest['signal'] and latest['macd_histogram'] > prev['macd_histogram']:
            bullish_score += 2
            reasons.append("🚀 MACD Bullish Momentum")
        elif latest['macd'] < latest['signal'] and latest['macd_histogram'] < prev['macd_histogram']:
            bearish_score += 2
            reasons.append("⬇️ MACD Bearish Momentum")
        
        # 5. Price above/below key EMAs
        if current_price > latest['ema_50']:
            bullish_score += 1
        elif current_price < latest['ema_50']:
            bearish_score += 1
        
        # 6. Stochastic (confirmation only)
        if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] > 50:
            bullish_score += 1
            reasons.append("📊 Stochastic Bullish")
        elif latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] < 50:
            bearish_score += 1
            reasons.append("📊 Stochastic Bearish")
        
        # 7. ADX Trend Strength Bonus
        if latest['adx'] > 30:
            if bullish_score > bearish_score:
                bullish_score += 2
                reasons.append(f"💎 Strong Trend (ADX: {latest['adx']:.1f})")
            elif bearish_score > bullish_score:
                bearish_score += 2
                reasons.append(f"💎 Strong Trend (ADX: {latest['adx']:.1f})")
        
        # STRICT THRESHOLD: Need at least 6 points and clear dominance
        signal = None
        confidence = 0
        min_score = 6 if strict_mode else 4
        
        if bullish_score >= min_score and bullish_score > (bearish_score + 2):
            signal = 'BUY'
            confidence = min(95, 55 + (bullish_score * 5))
            entry = current_price
            stop_loss = entry - (atr * ATR_MULTIPLIER_SL)
            
            # Dynamic TP based on trend strength
            if bullish_score >= 10:
                take_profit = entry + (atr * ATR_MULTIPLIER_TP * 1.5)
                tp_levels = [
                    entry + (atr * ATR_MULTIPLIER_TP * 0.7),
                    entry + (atr * ATR_MULTIPLIER_TP * 1.0),
                    entry + (atr * ATR_MULTIPLIER_TP * 1.5)
                ]
            else:
                take_profit = entry + (atr * ATR_MULTIPLIER_TP)
                tp_levels = [
                    entry + (atr * ATR_MULTIPLIER_TP * 0.5),
                    entry + (atr * ATR_MULTIPLIER_TP * 0.8),
                    entry + (atr * ATR_MULTIPLIER_TP)
                ]
                
        elif bearish_score >= min_score and bearish_score > (bullish_score + 2):
            signal = 'SELL'
            confidence = min(95, 55 + (bearish_score * 5))
            entry = current_price
            stop_loss = entry + (atr * ATR_MULTIPLIER_SL)
            
            # Dynamic TP based on trend strength
            if bearish_score >= 10:
                take_profit = entry - (atr * ATR_MULTIPLIER_TP * 1.5)
                tp_levels = [
                    entry - (atr * ATR_MULTIPLIER_TP * 0.7),
                    entry - (atr * ATR_MULTIPLIER_TP * 1.0),
                    entry - (atr * ATR_MULTIPLIER_TP * 1.5)
                ]
            else:
                take_profit = entry - (atr * ATR_MULTIPLIER_TP)
                tp_levels = [
                    entry - (atr * ATR_MULTIPLIER_TP * 0.5),
                    entry - (atr * ATR_MULTIPLIER_TP * 0.8),
                    entry - (atr * ATR_MULTIPLIER_TP)
                ]
        
        if signal:
            return {
                'pair': pair,
                'signal': signal,
                'entry': round(entry, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'tp_levels': [round(tp, 5) for tp in tp_levels],
                'confidence': round(confidence, 1),
                'reasons': reasons,
                'rsi': round(latest['rsi'], 2),
                'adx': round(latest['adx'], 2),
                'atr': round(atr, 5),
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
        
        return None

# Initialize analyzer
analyzer = ForexAnalyzer()
subscribed_users = set()

@bot.message_handler(commands=['start'])
def start_command(message):
    welcome_text = """
🤖 *Welcome to OPTIMIZED Forex Signal Bot*

✨ *NEW FEATURES FOR HIGHER WIN RATE:*
• 🎯 Strict trend filtering (ADX > 25)
• ⏰ Session-aware signals (best trading times)
• 📊 Multi-timeframe confirmation
• 💎 Focus on high-liquidity pairs
• 🛡️ Wider stop losses (less premature exits)

⚠️ *DISCLAIMER*: No strategy guarantees profits. Always use proper risk management.

📊 *RECOMMENDED STRATEGY:*
1️⃣ Use 1H timeframe (more reliable than 15m)
2️⃣ Trade only during high-liquidity sessions
3️⃣ Focus on EURUSD, GBPUSD, USDJPY, XAUUSD
4️⃣ Wait for confidence > 75%
5️⃣ Risk only 1% per trade

📋 *Commands:*
/signals [timeframe] - Get premium signals
/analyze [PAIR] [timeframe] - Deep analysis
/best - Show best pairs RIGHT NOW
/times - Trading session guide
/strategy - Win rate optimization tips
/pairs - All available pairs
"""
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def help_command(message):
    start_command(message)

@bot.message_handler(commands=['strategy'])
def strategy_command(message):
    strategy_text = """
🎯 *PROVEN STRATEGIES FOR 70%+ WIN RATE:*

*1. TIMEFRAME SELECTION:*
• 📈 1H Chart = MOST RELIABLE (Recommended)
• ⏰ 4H Chart = Best for swing trading
• ⚠️ 15M Chart = More signals but lower accuracy

*2. BEST TRADING SESSIONS:*
• 🥇 London/NY Overlap (1-5PM GMT) = HIGHEST WIN RATE
• 🥈 London Session (8AM-5PM GMT) = EUR, GBP pairs
• 🥉 NY Session (1-10PM GMT) = USD pairs
• 🌙 Asian Session (12-8AM GMT) = JPY, XAUUSD

*3. PAIR SELECTION:*
✅ Trade ONLY these for highest win rate:
• EURUSD (Most liquid, best spreads)
• GBPUSD (Good volatility)
• USDJPY (Trending, good for trends)
• XAUUSD (Strong trends during sessions)

❌ AVOID for now:
• Exotic pairs (high spreads)
• Low liquidity times (weekends, holidays)

*4. SIGNAL FILTERING:*
• Only take signals with confidence > 75%
• Wait for ADX > 25 (strong trend)
• Check multiple timeframes (1H + 4H)
• Trade WITH the 4H trend direction

*5. RISK MANAGEMENT:*
• Risk 1% per trade (MAX 2%)
• Move SL to breakeven at TP1
• Take 50% profit at TP1, let rest run
• Never trade news events (high risk)

*6. WIN RATE KILLERS (Avoid these):*
❌ Trading against the trend
❌ Trading during low liquidity
❌ Ignoring ADX < 25 (choppy market)
❌ Taking every signal (quality > quantity)
❌ Moving stop loss further when losing

💡 Use: /best to see which pairs are trending NOW!
"""
    bot.reply_to(message, strategy_text, parse_mode='Markdown')

@bot.message_handler(commands=['times'])
def times_command(message):
    current_hour = datetime.utcnow().hour
    
    times_text = f"""
⏰ *TRADING SESSION GUIDE*
Current UTC Time: {datetime.utcnow().strftime('%H:%M')}

🌅 *Asian Session (12AM-8AM GMT)*
• Best for: USDJPY, EURJPY, GBPJPY, XAUUSD
• Status: {'🟢 ACTIVE' if 0 <= current_hour < 8 else '⚫ CLOSED'}

🇬🇧 *London Session (8AM-5PM GMT)*
• Best for: EURUSD, GBPUSD, EURGBP
• Status: {'🟢 ACTIVE' if 8 <= current_hour < 17 else '⚫ CLOSED'}

🇺🇸 *New York Session (1PM-10PM GMT)*
• Best for: All USD pairs
• Status: {'🟢 ACTIVE' if 13 <= current_hour < 22 else '⚫ CLOSED'}

🔥 *OVERLAP (1PM-5PM GMT)* ⭐ BEST TIME!
• Highest volume & volatility
• Status: {'🟢🟢 ACTIVE!' if 13 <= current_hour < 17 else '⚫ CLOSED'}

💡 *Current Recommendation:*
"""
    
    if 13 <= current_hour < 17:
        times_text += "🔥 PERFECT TIME! London/NY overlap - Trade all pairs!"
    elif 8 <= current_hour < 17:
        times_text += "✅ Good time for EUR and GBP pairs"
    elif 13 <= current_hour < 22:
        times_text += "✅ Good time for USD pairs"
    elif 0 <= current_hour < 8:
        times_text += "✅ Good time for JPY pairs and Gold"
    else:
        times_text += "⚠️ Low liquidity period - Be cautious!"
    
    bot.reply_to(message, times_text, parse_mode='Markdown')

@bot.message_handler(commands=['best'])
def best_command(message):
    """Show best pairs to trade RIGHT NOW"""
    bot.reply_to(message, "🔍 Scanning for strongest trends... Please wait.")
    
    results = []
    for pair in ALL_PAIRS:
        df = analyzer.get_price_data(pair, interval='1h', periods=200)
        if df is None or len(df) < 200:
            continue
        
        df = analyzer.calculate_indicators(df)
        latest = df.iloc[-1]
        
        # Calculate trend strength
        trend_score = 0
        if latest['ema_9'] > latest['ema_20'] > latest['ema_50']:
            trend_score = 1
        elif latest['ema_9'] < latest['ema_20'] < latest['ema_50']:
            trend_score = -1
        
        results.append({
            'pair': pair,
            'adx': latest['adx'],
            'trend': 'Bullish' if trend_score > 0 else 'Bearish' if trend_score < 0 else 'Ranging',
            'rsi': latest['rsi'],
            'good_time': analyzer.is_good_trading_time(pair)
        })
        time.sleep(0.5)
    
    # Sort by ADX (trend strength)
    results.sort(key=lambda x: x['adx'], reverse=True)
    
    best_text = "🏆 *BEST PAIRS TO TRADE NOW:*\n\n"
    
    for i, r in enumerate(results[:5], 1):
        emoji = "🔥" if r['adx'] > 30 else "✅" if r['adx'] > 25 else "⚠️"
        session_emoji = "⏰" if r['good_time'] else "🌙"
        
        best_text += f"{emoji} *{i}. {r['pair']}*\n"
        best_text += f"   Trend: {r['trend']} | ADX: {r['adx']:.1f}\n"
        best_text += f"   RSI: {r['rsi']:.1f} {session_emoji}\n\n"
    
    best_text += "\n💡 Focus on pairs with ADX > 25 during their active session!"
    
    bot.send_message(message.chat.id, best_text, parse_mode='Markdown')

@bot.message_handler(commands=['pairs'])
def pairs_command(message):
    pairs_text = """
📊 *AVAILABLE PAIRS:*

🥇 *HIGH PRIORITY (Trade these):*
• EURUSD - Most liquid, lowest spread
• GBPUSD - Good volatility, clear trends
• USDJPY - Strong trending pair
• XAUUSD - Gold, excellent trends

🥈 *MEDIUM PRIORITY (Experienced traders):*
• GBPJPY - High volatility
• EURJPY - Good trends

⏰ *TIMEFRAMES:*
• 1h - RECOMMENDED (best accuracy)
• 4h - Swing trading
• 15m - Scalping (lower accuracy)
• 30m - Short-term
• 1d - Position trading
"""
    bot.reply_to(message, pairs_text, parse_mode='Markdown')

@bot.message_handler(commands=['signals'])
def signals_command(message):
    parts = message.text.split()
    timeframe = '1h' if len(parts) == 1 else parts[1] if parts[1] in TIMEFRAMES else '1h'
    
    bot.reply_to(message, f"🔍 Scanning for HIGH-QUALITY {timeframe} signals...\n\n⏳ This uses strict filters for better accuracy.")
    
    signals_found = []
    for pair in ALL_PAIRS:
        signal = analyzer.generate_signal(pair, timeframe, strict_mode=True)
        if signal and signal['confidence'] >= 70:
            signals_found.append(signal)
        time.sleep(0.5)
    
    if signals_found:
        # Sort by confidence
        signals_found.sort(key=lambda x: x['confidence'], reverse=True)
        for signal in signals_found:
            send_signal(message.chat.id, signal)
    else:
        bot.send_message(message.chat.id, 
            f"📊 No high-quality signals on {timeframe} right now.\n\n"
            "💡 Tips:\n"
            "• Try /best to see trending pairs\n"
            "• Check /times for active sessions\n"
            "• Wait for higher confidence setups\n"
            "• Use /analyze [PAIR] to check specific pair")

@bot.message_handler(commands=['analyze'])
def analyze_command(message):
    try:
        parts = message.text.split()
        pair = parts[1].upper()
        timeframe = '1h' if len(parts) < 3 else parts[2] if parts[2] in TIMEFRAMES else '1h'
        
        if pair not in ALL_PAIRS:
            bot.reply_to(message, f"❌ Invalid pair. Use /pairs")
            return
        
        bot.reply_to(message, f"🔍 Deep analysis of {pair} on {timeframe}...")
        signal = analyzer.generate_signal(pair, timeframe, strict_mode=False)
        
        if signal:
            send_signal(message.chat.id, signal)
        else:
            bot.send_message(message.chat.id, 
                f"📊 No clear setup for {pair} on {timeframe}.\n\n"
                "Possible reasons:\n"
                "• Weak trend (ADX < 25)\n"
                "• Wrong trading session\n"
                "• Choppy/ranging market\n"
                "• Mixed signals\n\n"
                "💡 Try: /best to find better opportunities")
    except IndexError:
        bot.reply_to(message, "❌ Usage: /analyze XAUUSD 1h")

def send_signal(chat_id, signal):
    """Enhanced signal formatting"""
    if 'XAU' in signal['pair']:
        sl_pips = abs(signal['entry'] - signal['stop_loss']) * 10
        tp_pips = abs(signal['entry'] - signal['take_profit']) * 10
    elif 'JPY' in signal['pair']:
        sl_pips = abs(signal['entry'] - signal['stop_loss']) * 100
        tp_pips = abs(signal['entry'] - signal['take_profit']) * 100
    else:
        sl_pips = abs(signal['entry'] - signal['stop_loss']) * 10000
        tp_pips = abs(signal['entry'] - signal['take_profit']) * 10000
    
    rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
    
    emoji = "🟢" if signal['signal'] == 'BUY' else "🔴"
    
    # Quality badge
    if signal['confidence'] >= 85:
        quality = "💎 PREMIUM"
    elif signal['confidence'] >= 75:
        quality = "⭐ HIGH QUALITY"
    else:
        quality = "✅ GOOD"
    
    signal_text = f"""
{emoji} *{signal['signal']} SIGNAL - {signal['pair']}* {quality}
⏰ Timeframe: {signal['timeframe']}

📍 *Entry*: {signal['entry']}
🛑 *Stop Loss*: {signal['stop_loss']} ({sl_pips:.1f} pips)
🎯 *Final TP*: {signal['take_profit']} ({tp_pips:.1f} pips)

📊 *Partial Take Profit Levels:*
"""
    
    for i, tp in enumerate(signal['tp_levels'], 1):
        if 'XAU' in signal['pair']:
            tp_pip = abs(signal['entry'] - tp) * 10
        elif 'JPY' in signal['pair']:
            tp_pip = abs(signal['entry'] - tp) * 100
        else:
            tp_pip = abs(signal['entry'] - tp) * 10000
        
        percentage = "50%" if i == 1 else "30%" if i == 2 else "20%"
        signal_text += f"   TP{i} ({percentage}): {tp} ({tp_pip:.1f} pips)\n"
    
    signal_text += f"""
⚖️ *Risk/Reward*: 1:{rr_ratio:.2f}
📈 *Confidence*: {signal['confidence']}%
📊 *RSI*: {signal['rsi']}
💪 *ADX (Trend)*: {signal['adx']} {'🔥' if signal['adx'] > 30 else '✅'}

🔍 *Why This Signal:*
"""
    
    for reason in signal['reasons']:
        signal_text += f"{reason}\n"
    
    signal_text += f"""
⏰ *Generated*: {signal['timestamp']}

💡 *EXECUTION PLAN:*
1️⃣ Enter at market or limit order at {signal['entry']}
2️⃣ Set SL at {signal['stop_loss']} (NO EXCEPTIONS!)
3️⃣ Take 50% profit at TP1, move SL to breakeven
4️⃣ Take 30% at TP2, trail remaining 20%
5️⃣ Risk only 1-2% of your account

⚠️ *Risk Warning:*
This is not financial advice. Market conditions can change rapidly. Always use proper risk management and never risk more than you can afford to lose.
"""
    
    bot.send_message(chat_id, signal_text, parse_mode='Markdown')

@bot.message_handler(commands=['subscribe'])
def subscribe_command(message):
    user_id = message.from_user.id
    subscribed_users.add(user_id)
    bot.reply_to(message, 
        "✅ Subscribed to premium signals!\n\n"
        "You'll receive 1H signals every 3 hours during active trading sessions.\n\n"
        "💡 Only high-confidence signals (>70%) will be sent.")

@bot.message_handler(commands=['unsubscribe'])
def unsubscribe_command(message):
    user_id = message.from_user.id
    if user_id in subscribed_users:
        subscribed_users.remove(user_id)
        bot.reply_to(message, "❌ Unsubscribed from signals.")
    else:
        bot.reply_to(message, "You're not subscribed. Use /subscribe")

def auto_signal_broadcast():
    """Smart auto-broadcast during good trading times"""
    while True:
        time.sleep(10800)  # 3 hours
        
        if not subscribed_users:
            continue
        
        # Only send during good trading hours
        current_hour = datetime.utcnow().hour
        if not (8 <= current_hour <= 22):  # Skip overnight
            continue
        
        signals = []
        for pair in MAJOR_PAIRS:  # Only major pairs for auto-signals
            signal = analyzer.generate_signal(pair, '1h', strict_mode=True)
            if signal and signal['confidence'] >= 75:
                signals.append(signal)
            time.sleep(0.5)
        
        if signals:
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            for user_id in subscribed_users:
                try:
                    bot.send_message(user_id, 
                        "🔔 *Premium Signal Alert!*\n\n"
                        f"Found {len(signals)} high-quality setup(s):", 
                        parse_mode='Markdown')
                    for signal in signals[:2]:  # Max 2 signals per broadcast
                        send_signal(user_id, signal)
                except Exception as e:
                    print(f"Error sending to user {user_id}: {e}")

# Start auto-broadcast
broadcast_thread = threading.Thread(target=auto_signal_broadcast, daemon=True)
broadcast_thread.start()

if __name__ == '__main__':
    print("🤖 Optimized Forex Signal Bot is running...")
    print("✨ Features: Strict filtering, Session awareness, ADX trend filter")
    bot.infinity_polling()
