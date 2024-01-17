from langchain.llms import VLLM

llm = VLLM(
    model="meta-llama/Llama-2-13b-hf",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,

)
for chunk in llm.stream("Write me a song about sparkling water in 10 paragraph."):
    print(chunk, end="", flush=True)