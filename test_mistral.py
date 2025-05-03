from llama_cpp import Llama

llm = Llama(
    model_path="D:/llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=512,
    n_gpu_layers=-1,  # Use GPU for all layers
    verbose=True
)

prompt = "You are a mysterious janitor. Player: Where is my friend?\nJanitor:"
output = llm(prompt, max_tokens=100, stop=["Player:", "Janitor:"], echo=False)

print("🧠 Janitor says:", output["choices"][0]["text"].strip())
