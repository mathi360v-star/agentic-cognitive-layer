import chromadb
from chromadb.utils import embedding_functions

class VectorVault:
    def __init__(self):
        # Ensure persistence across chunks if cached
        self.client = chromadb.PersistentClient(path="./dataset/chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="rca_memory", 
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def store_lesson(self, lesson_text, domain):
        import uuid
        self.collection.add(
            documents=[lesson_text],
            metadatas=[{"domain": domain}],
            ids=[f"rca_{uuid.uuid4().hex[:8]}"]
        )

    def retrieve_lessons(self, query, domain):
        try:
            res = self.collection.query(query_texts=[query], where={"domain": domain}, n_results=2)
            return res['documents'][0] if res['documents'] else []
        except: return []

vault = VectorVault()