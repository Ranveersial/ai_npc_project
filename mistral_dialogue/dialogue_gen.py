from llama_cpp import Llama
import random

# Load Mistral 7B
llm = Llama(
    model_path="D:/llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=768,
    n_gpu_layers=-1,
    verbose=False
)

# Memory and knowledge base
conversation_memory = []

janitor_knowledge = {
    "status": "She’s gone.",
    "guilt": "He didn’t stop it.",
    "involvement": "He saw it happen.",
    "confession": "He didn’t do it alone.",
}

def generate_dialogue(player_input: str) -> str:
    # Add player input to memory
    conversation_memory.append(f"Player: {player_input}")

    # System prompt and behavioral description
    system_prompt = (
        "You are 'The Janitor' – a cryptic, emotionally intelligent man who knows what happened to the player's missing friend.\n"
        "You live in an abandoned, haunted building and speak with unsettling calmness.\n"
        "You never directly admit guilt, but you hint at knowing everything.\n"
        "You often answer with emotional weight, sarcasm, or counter-questions.\n"
        "Your sentences are natural, varied, and realistic – like a person who’s hiding something painful.\n\n"
    )

    # Inject memory-based consistency
    janitor_memory = (
        f"The Janitor remembers this:\n"
        f"- The friend is missing. {janitor_knowledge['status']}\n"
        f"- He was present, but not directly guilty. {janitor_knowledge['guilt']}\n"
        f"- He feels uneasy. {janitor_knowledge['involvement']}\n"
        f"- He might be protecting someone. {janitor_knowledge['confession']}\n\n"
    )

    # Few-shot examples for tone and pacing
    few_shot = (
        "Player: Where is my friend?\n"
        "Janitor: Depends. Do you believe she's still alive?\n\n"
        "Player: I’m not sure.\n"
        "Janitor: Good. Certainty is dangerous in a place like this.\n\n"
        "Player: Did you see her?\n"
        "Janitor: I see many things. Some I regret. Some I try to forget.\n\n"
        "Player: Are you playing games with me?\n"
        "Janitor: No. The games already ended. You’re just late to find out who won.\n"
    )

    # Use last 3 exchanges for context
    recent_history = conversation_memory[-6:]
    full_prompt = system_prompt + janitor_memory + few_shot + "\n" + "\n".join(recent_history) + "\nJanitor:"

    # Generate response
    response = llm(
        full_prompt,
        max_tokens=100,
        stop=["Player:", "Janitor:"],
        echo=False
    )
    reply = response["choices"][0]["text"].strip()

    # Repeat protection with fallback
    if len(conversation_memory) >= 2:
        last_reply = conversation_memory[-2].split(":", 1)[-1].strip().lower()
        this_reply = reply.lower()

        if this_reply == last_reply:
            fallback_lines = [
                "You’re not ready to hear that.",
                "I’ve said too much already.",
                "The rest... isn’t for you to know.",
                "You wouldn’t believe me if I told you.",
                "[The janitor goes silent, looking at the floor.]"
            ]
            reply = random.choice(fallback_lines)

    # Store response in memory
    conversation_memory.append(f"Janitor: {reply}")
    return reply or "[⚠️ Janitor said nothing.]"
