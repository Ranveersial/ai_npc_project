import random
from rl_agent.janitor_env import JanitorTrustEnv
from stable_baselines3 import PPO
from mistral_dialogue.dialogue_gen import generate_dialogue
from textblob import TextBlob
from voice_handler.speak import speak  # Uncomment if using voice

# Load PPO model
ppo_model = PPO.load("janitor_smart_agent")

# Interrogation prompts
interrogation_questions = [
    "What did she fear the most?",
    "What did she never talk about?",
    "Tell me something only she knew.",
    "What made her stop smiling?",
    "What was her last secret?"
]

# NLP + keyword-based feature extractor
def extract_features_from_player_input(text: str):
    text = text.lower()
    confidence_keywords = ["definitely", "for sure", "she always", "never", "no doubt"]
    low_conf_keywords = ["maybe", "not sure", "i guess", "possibly", "i think"]

    # Sentiment-based modifier
    sentiment_score = TextBlob(text).sentiment.polarity  # Range: [-1, 1]
    sentiment_conf = 0.5 + (sentiment_score / 2)  # Normalize to [0,1]

    if any(k in text for k in confidence_keywords):
        confidence = max(0.8, sentiment_conf)
    elif any(k in text for k in low_conf_keywords):
        confidence = min(0.4, sentiment_conf)
    else:
        confidence = sentiment_conf

    memory_flag = 1.0 if "lie" in text or "not true" in text or "i made that up" in text else 0.0
    return [round(confidence, 2), memory_flag]

# Map action to trust/trap/doubt
def map_action_to_decision(action: int):
    return {0: "trust", 1: "trap", 2: "doubt"}.get(action, "neutral")

# Add behavioral tone to dialogue
def generate_janitor_response(player_input: str, decision: str):
    tag_line = {
        "trust": "[The Janitor seems to believe you. His voice softens.]",
        "trap": "[He tilts his head, as if he knows you're hiding something.]",
        "doubt": "[A long pause. You feel judged.]"
    }.get(decision, "")

    prompt = f"{player_input} {tag_line}"
    return generate_dialogue(prompt)

# 🎮 MAIN LOOP
print("\n🎮 Janitor Dialogue System: Free Talk + Interrogation Loop\n")
print("Type 'exit' to quit.\n")

turns = 0
state = "free_chat"

while True:
    if state == "free_chat":
        player_input = input("You: ")
        if player_input.lower() == "exit":
            break

        janitor_reply = generate_dialogue(player_input)
        print(f"Janitor: {janitor_reply}")
        speak(janitor_reply)

        turns += 1
        if turns >= 3:
            state = "interrogation"
            print("\n[The Janitor stops. His tone shifts.]\n")

    elif state == "interrogation":
        question = random.choice(interrogation_questions)
        print(f"Janitor: {question}")
        speak(question)
        player_answer = input("You: ")
        if player_answer.lower() == "exit":
            break

        features = extract_features_from_player_input(player_answer)
        action, _ = ppo_model.predict([features], deterministic=True)
        decision = map_action_to_decision(int(action))

        print(f"[Agent Decision → {decision.upper()} | Features: {features}]")

        janitor_reply = generate_janitor_response(player_answer, decision)
        print(f"Janitor: {janitor_reply}")
        speak(janitor_reply)

        turns = 0
        state = "free_chat"
