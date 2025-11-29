import os
import random
from stable_baselines3 import PPO
from mistral_dialogue.dialogue_gen import generate_dialogue
from textblob import TextBlob

# Voice (optional) — try both import styles; fall back to no-op
try:
    from voice_handler.speak import speak  # your original path
    VOICE_ENABLED = True
except Exception:
    try:
        from rl_agent.voice_handler.speak import speak
        VOICE_ENABLED = True
    except Exception:
        VOICE_ENABLED = False
        def speak(_):  # no-op
            pass

# --- PPO checkpoint loader (tries common paths) ---
def _load_ppo():
    candidates = [
        "janitor_smart_agent",                  # your existing name
        os.path.join("rl_agent", "janitor_agent.zip"),
        "janitor_smart_agent.zip"
    ]
    last_err = None
    for path in candidates:
        try:
            model = PPO.load(path)
            print(f"[PPO] Loaded checkpoint: {path}")
            return model
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load PPO checkpoint. Tried: {candidates}\nLast error: {last_err}")

ppo_model = _load_ppo()

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
    text_l = text.lower()
    hi_conf = ["definitely", "for sure", "she always", "never", "no doubt"]
    lo_conf = ["maybe", "not sure", "i guess", "possibly", "i think"]

    # Sentiment-based modifier
    try:
        sentiment_score = TextBlob(text_l).sentiment.polarity  # [-1, 1]
    except Exception:
        sentiment_score = 0.0
    sentiment_conf = 0.5 + (sentiment_score / 2.0)  # [0,1]

    if any(k in text_l for k in hi_conf):
        confidence = max(0.8, sentiment_conf)
    elif any(k in text_l for k in lo_conf):
        confidence = min(0.4, sentiment_conf)
    else:
        confidence = sentiment_conf

    memory_flag = 1.0 if any(kw in text_l for kw in ["lie", "not true", "i made that up"]) else 0.0
    return [round(confidence, 2), memory_flag]

# Map action to trust/trap/doubt
def map_action_to_decision(action: int):
    return {0: "trust", 1: "trap", 2: "doubt"}.get(action, "neutral")

# Add behavioral directive for interrogation tone
def generate_janitor_response(player_input: str, decision: str):
    directive = {
        "trust": "[INTERROGATION MODE | DECISION: TRUST | Give one small, concrete, grounded hint. No invented names or stories. Keep it to 1–2 sentences.]",
        "trap":  "[INTERROGATION MODE | DECISION: TRAP | Ask a pointed follow-up that forces a specific detail (time/place/door). 1 sentence.]",
        "doubt": "[INTERROGATION MODE | DECISION: DOUBT | Point out uncertainty or inconsistency and press for one detail. 1 sentence.]"
    }.get(decision, "[Stay in character.]")
    prompt = f"{directive}\n{player_input}".strip()
    return generate_dialogue(prompt)

# 🎮 MAIN LOOP
print("\n🎮 Rogan the Janitor: Free Talk + Interrogation Loop\n")
print("Type 'exit' to quit.\n")

turns = 0
state = "free_chat"

# Intro line (from character profile)
intro_line = generate_dialogue()  # no input = first_mes
print(f"Janitor: {intro_line}")
if VOICE_ENABLED:
    speak(intro_line)

while True:
    if state == "free_chat":
        player_input = input("You: ")
        if player_input.strip().lower() == "exit":
            break

        janitor_reply = generate_dialogue(player_input)
        print(f"Janitor: {janitor_reply}")
        if VOICE_ENABLED:
            speak(janitor_reply)

        turns += 1
        if turns >= 3:
            state = "interrogation"
            print("\n[Rogan stops. His tone shifts.]\n")

    elif state == "interrogation":
        question = random.choice(interrogation_questions)
        print(f"Janitor: {question}")
        if VOICE_ENABLED:
            speak(question)

        player_answer = input("You: ")
        if player_answer.strip().lower() == "exit":
            break

        features = extract_features_from_player_input(player_answer)
        action, _ = ppo_model.predict([features], deterministic=True)

        # Safe scalar extraction (no NumPy deprecation)
        try:
            action_scalar = int(action.item())
        except Exception:
            action_scalar = int(action[0])

        decision = map_action_to_decision(action_scalar)
        print(f"[Agent Decision → {decision.upper()} | Features: {features}]")

        janitor_reply = generate_janitor_response(player_answer, decision)
        print(f"Janitor: {janitor_reply}")
        if VOICE_ENABLED:
            speak(janitor_reply)

        turns = 0
        state = "free_chat"
