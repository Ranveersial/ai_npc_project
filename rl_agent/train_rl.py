from stable_baselines3 import PPO
from rl_agent.janitor_env import JanitorTrustEnv

# Create environment
env = JanitorTrustEnv()

# Create PPO agent with MLP policy
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=150000)  # More steps for better convergence

# Save the trained model
model.save("janitor_smart_agent")

print("✅ Smart Janitor agent trained and saved as 'janitor_smart_agent.zip'")
