import pyttsx3

# Initialize engine only once
engine = pyttsx3.init()
engine.setProperty('rate', 120)         # Slower pace for suspense
engine.setProperty('volume', 1.0)       # Max volume

# Select a deeper/male voice if available
voices = engine.getProperty('voices')
selected = None
for v in voices:
    if "male" in v.name.lower() or "baritone" in v.name.lower():
        selected = v.id
        break

# Fallback if no deep male found
if selected is None and voices:
    selected = voices[0].id

engine.setProperty('voice', selected)

def speak(text):
    engine.say(text)
    engine.runAndWait()
