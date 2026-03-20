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

# THE TOKEN BUCKET
RATE_LIMITS = {
    "groq": 2.1,       
    "google": 4.1,     
    "cohere": 2.1,
    "cerebras": 1.1,
    "sambanova": 1.1,
    "mistral": 2.1
}

LAST_CALL_TIME = {k: 0.0 for k in RATE_LIMITS}
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

    # THE FIX: Store the LLM and the provider tag safely inside a dictionary
    for key in groq_keys: 
        llm = ChatGroq(api_key=key, model_name="llama-3.3-70b-versatile", max_retries=0)
        pool.append({"llm": llm, "provider": "groq"})
        
    for key in google_keys: 
        llm = ChatGoogleGenerativeAI(api_key=key, model="gemini-2.0-flash", max_retries=0)
        pool.append({"llm": llm, "provider": "google"})
        
    for key in cohere_keys: 
        llm = ChatCohere(cohere_api_key=key, model="command-r-plus", max_retries=0)
        pool.append({"llm": llm, "provider": "cohere"})
        
    for key in cerebras_keys: 
        llm = ChatOpenAI(api_key=key, base_url="https://api.cerebras.ai/v1", model_name="llama-3.3-70b", max_retries=0)
        pool.append({"llm": llm, "provider": "cerebras"})
        
    for key in sambanova_keys: 
        llm = ChatOpenAI(api_key=key, base_url="https://api.sambanova.ai/v1", model_name="Meta-Llama-3.1-8B-Instruct", max_retries=0)
        pool.append({"llm": llm, "provider": "sambanova"})
        
    for key in mistral_keys: 
        llm = ChatMistralAI(api_key=key, model="mistral-small-latest", max_retries=0)
        pool.append({"llm": llm, "provider": "mistral"})
        
    return pool

GLOBAL_LLM_POOL = build_llm_pool()

# =====================================================================
# LANE 1: THE FAST LANE (Uses all 30 keys for general tasks)
# =====================================================================
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=16))
async def safe_async_invoke(messages: list, temperature: float = 0.2) -> str:
    if not GLOBAL_LLM_POOL:
        raise ValueError("System Error: No LLMs initialized.")

    current_cascade = list(GLOBAL_LLM_POOL)
    random.shuffle(current_cascade)

    last_exception = None

    for item in current_cascade:
        llm = item["llm"]
        provider = item["provider"]
        
        # --- THE DRIP SYSTEM (Token Bucket Algorithm) ---
        async with LOCK:
            now = time.time()
            time_since_last_call = now - LAST_CALL_TIME[provider]
            required_cooldown = RATE_LIMITS[provider]
            
            if time_since_last_call < required_cooldown:
                sleep_time = required_cooldown - time_since_last_call
                await asyncio.sleep(sleep_time)
            
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

# =====================================================================
# LANE 2: THE HEAVY LANE (Uses only Frontier Models for Strict Judging)
# =====================================================================
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=16))
async def heavy_async_invoke(messages: list, temperature: float = 0.0) -> str:
    """ONLY routes to 70B+ or frontier models for complex mathematical judging."""
    if not GLOBAL_LLM_POOL:
        raise ValueError("System Error: No LLMs initialized.")

    # Filter the pool to ONLY include the high-IQ heavyweights
    heavy_pool = [item for item in GLOBAL_LLM_POOL if item["provider"] in ["groq", "google", "cohere", "cerebras"]]
    
    if not heavy_pool:
        raise ValueError("System Error: No heavy LLMs available in the pool.")

    current_cascade = list(heavy_pool)
    random.shuffle(current_cascade)

    last_exception = None

    for item in current_cascade:
        llm = item["llm"]
        provider = item["provider"]
        
        # --- THE DRIP SYSTEM (Token Bucket Algorithm) ---
        async with LOCK:
            now = time.time()
            time_since_last_call = now - LAST_CALL_TIME[provider]
            required_cooldown = RATE_LIMITS[provider]
            
            if time_since_last_call < required_cooldown:
                sleep_time = required_cooldown - time_since_last_call
                await asyncio.sleep(sleep_time)
            
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
            print(f"[!] HEAVY ROUTER: {model_name} failed. EXACT ERROR: {error_msg}...")
            last_exception = e
            continue 

    print("[-] Heavy Swarm overload. Triggering exponential backoff sleep...")
    raise last_exception