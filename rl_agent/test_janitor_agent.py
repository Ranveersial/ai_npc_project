from stable_baselines3 import PPO
from rl_agent.janitor_env import JanitorTrustEnv

# Load smart agent
model = PPO.load("janitor_smart_agent")

# Create environment
env = JanitorTrustEnv()

# Run test episodes
for episode in range(10):
    obs, _ = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        action_meaning = {0: "Answer A", 1: "Answer B", 2: "Answer C"}
        result = "Trusted" if reward > 0 else "Trapped"
        print(f"Action Taken: {action_meaning[int(action)]} | Outcome: {result} | Reward: {reward}")

    print("-" * 60)

env.close()
