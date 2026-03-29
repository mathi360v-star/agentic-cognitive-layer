import os, random, asyncio, time
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from tenacity import retry, stop_after_attempt, wait_exponential

class ShardedRouter:
    def __init__(self, chunk_id: int, total_chunks: int):
        self.chunk_id = chunk_id
        self.total_chunks = total_chunks
        self.pool = self._build_shard()
        self.locks = {p: asyncio.Lock() for p in ["groq", "google", "cohere"]}
        self.last_call = {p: 0.0 for p in ["groq", "google", "cohere"]}

    def _get_shard(self, env_var):
        keys = [k.strip() for k in os.getenv(env_var, "").split(",") if k.strip()]
        if not keys: return []
        # Distribute keys across chunks
        size = max(1, len(keys) // self.total_chunks)
        start = self.chunk_id * size
        return keys[start : start + size]

    def _build_shard(self):
        shard = []
        for k in self._get_shard("GROQ_API_KEYS"):
            shard.append({"llm": ChatGroq(api_key=k, model_name="llama-3.3-70b-versatile"), "provider": "groq"})
        for k in self._get_shard("GOOGLE_API_KEYS"):
            shard.append({"llm": ChatGoogleGenerativeAI(api_key=k, model="gemini-2.0-flash"), "provider": "google"})
        for k in self._get_shard("COHERE_API_KEYS"):
            shard.append({"llm": ChatCohere(cohere_api_key=k, model="command-r-plus"), "provider": "cohere"})
        return shard

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=20))
    async def invoke(self, messages, temperature=0.2, heavy=False):
        # Heavy lane uses only 70B+ models
        candidates = [i for i in self.pool if not heavy or i["provider"] in ["groq", "google", "cohere"]]
        if not candidates: candidates = self.pool # Fallback
        random.shuffle(candidates)
        
        for item in candidates:
            async with self.locks[item["provider"]]:
                # STRICT COOLDOWN: 6 seconds for heavy, 3 seconds for safe
                cooldown = 6.0 if heavy else 3.0
                elapsed = time.time() - self.last_call[item["provider"]]
                if elapsed < cooldown:
                    await asyncio.sleep(cooldown - elapsed + random.uniform(0.5, 1.5))
                
                try:
                    res = await item["llm"].ainvoke(messages, temperature=temperature)
                    self.last_call[item["provider"]] = time.time()
                    return res.content
                except Exception as e:
                    if "429" in str(e):
                        print(f"[!] Shard {self.chunk_id}: {item['provider']} Rate Limited. Retrying with next key...")
                        continue
                    raise e
        raise RuntimeError(f"Shard {self.chunk_id}: All keys exhausted or 429.")