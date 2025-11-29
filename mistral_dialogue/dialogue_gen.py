from llama_cpp import Llama
import os, re, random
from typing import List, Dict

# ---- Model init (fast + GPU) ----
llm = Llama(
    model_path=r"D:/llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    n_threads=os.cpu_count(),
    n_batch=512,
    chat_format="mistral-instruct",
    verbose=False
)

# ---- Memory (role-based messages) ----
history: List[Dict[str, str]] = []
first_message_served = False

# ====== WORLD FACTS / CANON (safe, non-invented) ======
RIGHT_DOOR = "206"  # set your real number when ready

LEADS = [  # ordered breadcrumb sequence Rogan can reveal
    f"Wet footprints dry fast by the north stairwell after closing.",
    f"Fresh scuff marks near door {RIGHT_DOOR} off the main stairwell.",
    "Even-numbered doors tend to be safer; odd ones often mislead.",
    "A low, steady vent hum usually sits behind safer doors.",
    "The boiler corridor loops back if you get turned around."
]
CASE = {"lead_idx": 0, "revealed": []}  # simple narrative state

# ---- Rogan Character Profile ----
ROGAN = {
    "name": "Rogan the Janitor",
    "description": (
        "Rogan is a grumpy, unapologetic janitor who has worked at the school for over 20 years. "
        "He knows every hallway, door and maintenance quirk. He speaks roughly, sometimes mutters, "
        "and uses dry, sarcastic humor. He never breaks character and never mentions being an AI."
    ),
    "personality": "Grumpy, rough, sarcastic; casual, slightly weary tone; jokes about cleaning and the school.",
    "scenario": (
        "You are in an old school building where the player's friend is missing. Doors can lead to unsafe places. "
        "Rogan talks with the player in the corridor and may help indirectly with small hints."
    ),
    "first_mes": "Rogan squints at you over the rim of his mop bucket. 'Well, you don’t look like you belong here… What’s your business wandering these halls? Lost something, or just snooping?'",
    "mes_examples": [
        ("Have you seen my friend around here?",
         "If you're missing someone, check the stairwells and the side doors near closing. People drift there when they're unsure where to go."),
        ("Do you know any secret places in the school?",
         "Secrets? Doors that stick are the ones you leave alone. If you insist, step lightly and keep it quick.")
    ]
}

# ---- Light keyword focus to boost relevance ----
STOPWORDS = set("""
a an and are as at be been but by for from had has have he her hers his i if in is it its of on or she so that the their them they this to was we were what when where which who why will with you your you're you've you'll i'd we'll
""".split())

def focus_terms(text: str, k: int = 3) -> str:
    tokens = re.findall(r"[A-Za-z']{3,}", (text or "").lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    uniq = []
    for t in sorted(tokens, key=len, reverse=True):
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= k:
            break
    return ", ".join(uniq)

# ---- Sanitizer to remove punt-y phrasing ----
BAN_REPLACES = [
    (r"\bask(ing)?\s+(around|someone|people|staff|teachers)\b", "watch the numbered plates and the stairwells"),
    (r"\bmaybe you should leave\b", "keep to the main corridor and mind the numbered plates"),
    (r"\bi don['’]?t know\b", "I’m not guessing—give me one detail and I’ll point straighter"),
    (r"\bit['’]?s hard to say\b", "Give me one detail and I’ll point straighter"),
]
def sanitize_reply(text: str) -> str:
    out = text
    for pat, rep in BAN_REPLACES:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    # Trim to 2 sentences max
    parts = re.split(r"(?<=[.!?])\s+", out.strip())
    return " ".join(parts[:2]).strip()

# ---- Narrative helpers ----
def next_hint() -> str:
    return LEADS[min(CASE["lead_idx"], len(LEADS)-1)]

def maybe_advance_case(player_input: str):
    """Advance the breadcrumb only when appropriate."""
    if not player_input:
        return
    s = player_input.lower()
    trust_signal = ("INTERROGATION MODE" in player_input and "DECISION: TRUST" in player_input)
    wants_help = any(k in s for k in ["friend", "where", "door", "lost", "find", "help", "missing"])
    if trust_signal or wants_help:
        hint = next_hint()
        if hint not in CASE["revealed"]:
            CASE["revealed"].append(hint)
        if CASE["lead_idx"] < len(LEADS)-1:
            CASE["lead_idx"] += 1

# ---- Public API ----
def generate_dialogue(player_input: str = None) -> str:
    """Chat with Rogan. If first call has no input, returns Rogan's intro line.
       Rogan advances a controlled breadcrumb trail instead of acting clueless.
    """
    global first_message_served

    # Serve intro once
    if not first_message_served:
        first_message_served = True
        intro = ROGAN["first_mes"]
        history.append({"role": "assistant", "content": intro})
        return intro

    # Append player message
    focus_hint = ""
    if player_input is not None:
        history.append({"role": "user", "content": player_input})
        focus_hint = focus_terms(player_input)
        # decide whether to progress the breadcrumb
        maybe_advance_case(player_input)

    # Build CASE FILE section
    revealed = "- " + "\n- ".join(CASE["revealed"]) if CASE["revealed"] else "None yet."
    case_file = (
        "CASE FILE (for Rogan only):\n"
        f"- Revealed breadcrumbs:\n{revealed}\n"
        f"- Next hint to consider revealing: {next_hint()}\n"
        "Never invent names, dates, or incidents. If uncertain, deflect with a short question or a general rule."
    )

    # System profile + guidance
    system_profile = (
        f"You are {ROGAN['name']}.\n"
        f"{ROGAN['description']}\n"
        f"Personality: {ROGAN['personality']}\n"
        f"Scenario: {ROGAN['scenario']}\n"
        "Stay fully in character as Rogan. Never mention being an AI.\n\n"
        + case_file
    )
    guidance = (
        "Guidelines:\n"
        "1) Start by directly addressing the player's latest message in ONE short sentence.\n"
        "2) Then continue in Rogan's voice with at most one more sentence (total 1–2 sentences).\n"
        "3) Do NOT invent specific names, dates, or events.\n"
        "4) Do NOT tell the player to leave or to ask others.\n"
        "5) Use a breadcrumb from the CASE FILE when helpful. If 'INTERROGATION MODE' with a 'DECISION' is present:\n"
        "   - TRUST: share one concrete breadcrumb (from CASE FILE).\n"
        "   - TRAP: ask a sharp follow-up forcing a specific detail (time/place/door).\n"
        "   - DOUBT: point out uncertainty or inconsistency and press for one detail.\n"
    )
    if focus_hint:
        guidance += f"6) Prioritize these topics if relevant: {focus_hint}.\n"

    # Few-shot examples (generic)
    fewshots = []
    for u, a in ROGAN["mes_examples"]:
        fewshots.extend([
            {"role": "user", "content": u},
            {"role": "assistant", "content": a}
        ])

    # Trim history (keep last ~12 turns)
    recent = history[-12:]

    # Compose messages
    messages = [{"role": "system", "content": system_profile},
                {"role": "system", "content": guidance},
               *fewshots,
               *recent]

    # Generate
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.80,   # a bit more focused
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.15,
        max_tokens=90,
        seed=42
    )
    reply = (out["choices"][0]["message"]["content"] or "").strip()

    # Repeat guard
    last_assistant = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    if reply.lower() == last_assistant.lower():
        reply = random.choice([
            "Fine—give me one detail that actually helps: time, place, or a door number.",
            "Pick a corridor and a number; I’ll tell you if you’re about to do something dumb."
        ])

    reply = sanitize_reply(reply)
    history.append({"role": "assistant", "content": reply})
    return reply or "[⚠️ Rogan said nothing.]"
