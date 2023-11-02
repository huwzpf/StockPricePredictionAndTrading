from gym_trading_env.downloader import download
import datetime
import pandas as pd
import gymnasium as gym
import time
from stable_baselines3 import PPO
import gym_trading_env
# this is simple example based on https://gym-trading-env.readthedocs.io/en/latest/rl_tutorial.html

def train_PPO(env_train, model_name, timesteps=500000):
    """PPO model"""

    start = time.time()
    #model = PPO('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    model = PPO('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    # model.save(f"{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

# Download BTC/USDT historical data from Binance and stores it to directory ./data/binance-BTCUSDT-1h.pkl
download(exchange_names = ["binance"],
    symbols= ["BTC/USDT"],
    timeframe= "1h",
    dir = "data",
    since= datetime.datetime(year= 2020, month= 1, day=1),
)
# Import your fresh data
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")

# df is a DataFrame with columns : "open", "high", "low", "close", "Volume USD"

# Create the feature : ( close[t] - close[t-1] )/ close[t-1]
df["feature_close"] = df["close"].pct_change()

# Create the feature : open[t] / close[t]
df["feature_open"] = df["open"]/df["close"]

# Create the feature : high[t] / close[t]
df["feature_high"] = df["high"]/df["close"]

# Create the feature : low[t] / close[t]
df["feature_low"] = df["low"]/df["close"]

 # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()

df.dropna(inplace= True) # Clean again !
# Eatch step, the environment will return 5 inputs  : "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"

env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [0, 0.5, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
    )


ppo_model = train_PPO(env, "PPO_model_test")
done, truncated = False, False
observation, info = env.reset()
while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    position_index = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
    observation, reward, done, truncated, info = env.step(position_index)