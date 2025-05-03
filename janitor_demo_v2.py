from mistral_dialogue.dialogue_gen import generate_dialogue
import pyttsx3

print("🛠️ Script started.")

# Optional: enable or disable voice output
enable_voice = True
if enable_voice:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

# 🚫 Start with empty memory
chat_history = []

print("✅ Janitor is ready. Speak to him below.\n")
print("🎮 Welcome to the Janitor’s Room (Cinematic AI Mode)")
print("Talk to him — he will respond with memory and voice.")
print("(Type 'exit' to quit.)")
print("=" * 60)

while True:
    player_input = input("Player: ").strip()
    if player_input.lower() in ["exit", "quit"]:
        break

    chat_history.append(f"Player: {player_input}")

    # Keep only last 3 exchanges (6 lines)
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]

    # Prompt for Janitor to respond in-character
    prompt = "\n".join(chat_history) + "\nJanitor: "

    print("🧠 Janitor is thinking...\n")
    reply = generate_dialogue(prompt).strip()
    reply = reply.split("Player:")[0].strip()

    chat_history.append(f"Janitor: {reply}")

    print(f"\nJanitor: {reply}")
    print("-" * 60)

    if enable_voice:
        engine.say(reply)
        engine.runAndWait()
