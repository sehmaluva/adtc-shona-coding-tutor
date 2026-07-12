from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="./models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
    n_ctx=2048,      # context window
    n_threads=4,     # adjust based on your CPU cores
    verbose=False
)

# Test prompt
prompt = "How do I sort numbers from smallest to largest in Python? Explain briefly."

output = llm(
    f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
    max_tokens=300,
    stop=["<|end|>"],
    echo=False
)

print(output["choices"][0]["text"])
