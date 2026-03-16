import os
import random
import asyncio
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# --------------------------------------------------------------
# 1. BUILD THE MASSIVE GLOBAL POOL
# --------------------------------------------------------------
def build_llm_pool() -> list:
    """Detects comma-separated API keys and builds a distributed LLM pool."""
    pool = []

    groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
    google_keys = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
    cohere_keys = [k.strip() for k in os.getenv("COHERE_API_KEYS", "").split(",") if k.strip()]
    cerebras_keys = [k.strip() for k in os.getenv("CEREBRAS_API_KEYS", "").split(",") if k.strip()]

    for key in groq_keys:
        pool.append(ChatGroq(api_key=key, model_name="llama3-70b-8192"))
    for key in google_keys:
        pool.append(ChatGoogleGenerativeAI(api_key=key, model="gemini-1.5-flash"))
    for key in cohere_keys:
        pool.append(ChatCohere(cohere_api_key=key, model="command-r"))
    for key in cerebras_keys:
        pool.append(ChatOpenAI(api_key=key, base_url="https://api.cerebras.ai/v1", model_name="llama3.1-70b"))
        
    return pool

GLOBAL_LLM_POOL = build_llm_pool()

# --------------------------------------------------------------
# 2. THE LOAD-BALANCING ROUTER
# --------------------------------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=16))
async def safe_async_invoke(messages: list, temperature: float = 0.2) -> str:
    """
    Indestructible dynamic router. Shuffles the LLM pool on every call.
    Instantly cascades through backup models if the primary fails.
    """
    if not GLOBAL_LLM_POOL:
        raise ValueError("System Error: No LLMs initialized. Check your CI/CD Secrets.")

    current_cascade = list(GLOBAL_LLM_POOL)
    random.shuffle(current_cascade)

    last_exception = None

    for llm in current_cascade:
        try:
            bound_llm = llm.bind(temperature=temperature)
            response = await bound_llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            # Safely extract the model name without touching secure SecretStr objects
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'Unknown Model'))
            print(f"[!] {model_name} rate-limited or failed. Rerouting instantly...")
            last_exception = e
            continue 

    print("[-] Swarm overload: ALL models in the cascade failed. Triggering exponential backoff sleep...")
    raise last_exception