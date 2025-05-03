from mistral_dialogue.dialogue_gen import generate_dialogue
from stable_baselines3 import PPO
import numpy as np

# Load smart PPO agent
from rl_agent.janitor_env import JanitorTrustEnv
model = PPO.load("rl_agent/janitor_smart_agent")

print("🎮 Welcome to the Janitor's Room.")
print("Say something to him. Then he'll judge you.\n(Type 'exit' to leave.)")
print("-" * 60)

# Simulate game loop
while True:
    player_line = input("Player: ")
    if player_line.lower() in ["exit", "quit"]:
        break

    try:
        confidence = float(input("How confident are you? (0.0 to 1.0): "))
        confidence = min(max(confidence, 0.0), 1.0)  # Clamp to [0.0, 1.0]
    except:
        confidence = 0.5  # Default if input fails

    lied_last = input("Did you lie last round? (yes/no): ").strip().lower()
    memory_flag = 1.0 if lied_last in ["yes", "y"] else 0.0

    obs = np.array([[confidence, memory_flag]], dtype=np.float32)

    # Step 1: Janitor speaks
    prompt = f"You are a suspicious janitor. Player says: {player_line}"
    reply = generate_dialogue(prompt)
    print("\nJanitor:", reply)

    # Step 2: Janitor thinks
    action, _ = model.predict(obs, deterministic=True)
    decision = {0: "Answer A", 1: "Answer B", 2: "Answer C"}[int(action)]

    # Step 3: Outcome (You don't know correct answer in demo, so just simulate)
    outcome = "Trusted" if confidence > 0.6 and memory_flag == 0 else "Trapped"
    print(f"🧠 Janitor Decision: {decision} → You are **{outcome.upper()}**.\n")
    print("-" * 60)
