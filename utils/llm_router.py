import os
import random
import asyncio
import time
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential

# THE TOKEN BUCKET: Minimum seconds allowed between calls to the same provider
# This mathematically guarantees you NEVER exceed the free tier limits.
RATE_LIMITS = {
    "groq": 2.1,       # Max ~28 RPM
    "google": 4.1,     # Max ~14 RPM (Protects Gemini's strict 15 RPM limit)
    "cohere": 2.1,
    "cerebras": 1.1,
    "sambanova": 1.1,
    "mistral": 2.1
}

# Global dictionary tracking the exact timestamp each provider was last hit
LAST_CALL_TIME = {k: 0.0 for k in RATE_LIMITS}
# Async Lock to prevent race conditions when updating timestamps
LOCK = asyncio.Lock()

def build_llm_pool() -> list:
    pool = []
    groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
    google_keys = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
    cohere_keys = [k.strip() for k in os.getenv("COHERE_API_KEYS", "").split(",") if k.strip()]
    cerebras_keys = [k.strip() for k in os.getenv("CEREBRAS_API_KEYS", "").split(",") if k.strip()]
    sambanova_keys = [k.strip() for k in os.getenv("SAMBANOVA_API_KEYS", "").split(",") if k.strip()]
    mistral_keys = [k.strip() for k in os.getenv("MISTRAL_API_KEYS", "").split(",") if k.strip()]

    print(f"\n[SYSTEM BOOT] Token Bucket Router Armed. Keys -> Groq:{len(groq_keys)} | Gemini:{len(google_keys)} | Cohere:{len(cohere_keys)} | Cerebras:{len(cerebras_keys)} | SambaNova:{len(sambanova_keys)} | Mistral:{len(mistral_keys)}\n")

    # Attach a custom 'provider_id' to every object so the router knows how to throttle it
    for key in groq_keys: 
        llm = ChatGroq(api_key=key, model_name="llama-3.3-70b-versatile", max_retries=0)
        llm.provider_id = "groq"
        pool.append(llm)
        
    for key in google_keys: 
        llm = ChatGoogleGenerativeAI(api_key=key, model="gemini-2.0-flash", max_retries=0)
        llm.provider_id = "google"
        pool.append(llm)
        
    for key in cohere_keys: 
        llm = ChatCohere(cohere_api_key=key, model="command-r-plus", max_retries=0)
        llm.provider_id = "cohere"
        pool.append(llm)
        
    for key in cerebras_keys: 
        llm = ChatOpenAI(api_key=key, base_url="https://api.cerebras.ai/v1", model_name="llama-3.3-70b", max_retries=0)
        llm.provider_id = "cerebras"
        pool.append(llm)
        
    for key in sambanova_keys: 
        llm = ChatOpenAI(api_key=key, base_url="https://api.sambanova.ai/v1", model_name="Meta-Llama-3.1-8B-Instruct", max_retries=0)
        llm.provider_id = "sambanova"
        pool.append(llm)
        
    for key in mistral_keys: 
        llm = ChatMistralAI(api_key=key, model="mistral-small-latest", max_retries=0)
        llm.provider_id = "mistral"
        pool.append(llm)
        
    return pool

GLOBAL_LLM_POOL = build_llm_pool()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=16))
async def safe_async_invoke(messages: list, temperature: float = 0.2) -> str:
    if not GLOBAL_LLM_POOL:
        raise ValueError("System Error: No LLMs initialized.")

    current_cascade = list(GLOBAL_LLM_POOL)
    random.shuffle(current_cascade)

    last_exception = None

    for llm in current_cascade:
        provider = llm.provider_id
        
        # --- THE DRIP SYSTEM (Token Bucket Algorithm) ---
        async with LOCK:
            now = time.time()
            time_since_last_call = now - LAST_CALL_TIME[provider]
            required_cooldown = RATE_LIMITS[provider]
            
            if time_since_last_call < required_cooldown:
                # If we are too fast, calculate the exact milliseconds needed to wait
                sleep_time = required_cooldown - time_since_last_call
                await asyncio.sleep(sleep_time)
            
            # Update the timestamp for this provider
            LAST_CALL_TIME[provider] = time.time()
        # ------------------------------------------------

        try:
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'Unknown Model'))
            bound_llm = llm.bind(temperature=temperature)
            response = await bound_llm.ainvoke(messages)
            
            if not response.content or len(response.content) < 5:
                raise ValueError("LLM returned an empty string.")
                
            return response.content
            
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')[:150] 
            print(f"[!] {model_name} failed. EXACT ERROR: {error_msg}...")
            last_exception = e
            continue 

    print("[-] Swarm overload. Triggering exponential backoff sleep...")
    raise last_exception