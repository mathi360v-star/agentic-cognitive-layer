import os
import random
import asyncio
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential

def build_llm_pool() -> list:
    pool = []

    # Parse keys, stripping any hidden whitespace
    groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
    google_keys = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
    cohere_keys = [k.strip() for k in os.getenv("COHERE_API_KEYS", "").split(",") if k.strip()]
    cerebras_keys = [k.strip() for k in os.getenv("CEREBRAS_API_KEYS", "").split(",") if k.strip()]
    sambanova_keys = [k.strip() for k in os.getenv("SAMBANOVA_API_KEYS", "").split(",") if k.strip()]
    mistral_keys = [k.strip() for k in os.getenv("MISTRAL_API_KEYS", "").split(",") if k.strip()]

    print(f"\n[SYSTEM BOOT] Loaded Keys -> Groq:{len(groq_keys)} | Gemini:{len(google_keys)} | Cohere:{len(cohere_keys)} | Cerebras:{len(cerebras_keys)} | SambaNova:{len(sambanova_keys)} | Mistral:{len(mistral_keys)}\n")

    # -----------------------------------------------------------------------
    # CRITICAL FIX: UPDATED TO THE LATEST STABLE API MODEL STRINGS 
    # -----------------------------------------------------------------------
    for key in groq_keys: 
        # Groq decommissioned 3.0. We use Llama 3.3 70B Versatile
        pool.append(ChatGroq(api_key=key, model_name="llama-3.3-70b-versatile", max_retries=0))
        
    for key in google_keys: 
        # Google shifted 1.5. We use Gemini 2.0 Flash for massive rate limits
        pool.append(ChatGoogleGenerativeAI(api_key=key, model="gemini-2.0-flash", max_retries=0))
        
    for key in cohere_keys: 
        # Cohere's stable endpoint
        pool.append(ChatCohere(cohere_api_key=key, model="command-r-plus", max_retries=0))
        
    for key in cerebras_keys: 
        # Reverted to 3.1 for Cerebras stability
        pool.append(ChatOpenAI(api_key=key, base_url="https://api.cerebras.ai/v1", model_name="llama3.1-70b", max_retries=0))
        
    for key in sambanova_keys: 
        # SambaNova's fast 8B instruct model is much more stable than their 70B
        pool.append(ChatOpenAI(api_key=key, base_url="https://api.sambanova.ai/v1", model_name="Meta-Llama-3.1-8B-Instruct", max_retries=0))
        
    for key in mistral_keys: 
        # Mistral Small is much faster and less prone to rate limits than Large
        pool.append(ChatMistralAI(api_key=key, model="mistral-small-latest", max_retries=0))
        
    return pool

GLOBAL_LLM_POOL = build_llm_pool()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=16))
async def safe_async_invoke(messages: list, temperature: float = 0.2) -> str:
    if not GLOBAL_LLM_POOL:
        raise ValueError("System Error: No LLMs initialized. GitHub Secrets are completely empty.")

    current_cascade = list(GLOBAL_LLM_POOL)
    random.shuffle(current_cascade)

    last_exception = None

    for llm in current_cascade:
        model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'Unknown Model'))
        try:
            bound_llm = llm.bind(temperature=temperature)
            response = await bound_llm.ainvoke(messages)
            
            # The Professor Agent crash ("unterminated string literal") happens when an LLM outputs blank text.
            # This protects the system from blank outputs.
            if not response.content or len(response.content) < 5:
                raise ValueError("LLM returned an empty or malformed string.")
                
            return response.content
            
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')[:150] 
            print(f"[!] {model_name} failed. EXACT ERROR: {error_msg}...")
            last_exception = e
            continue 

    print("[-] Swarm overload. Triggering exponential backoff sleep...")
    raise last_exception