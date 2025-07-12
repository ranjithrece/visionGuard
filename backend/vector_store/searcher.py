from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class SaftyRetriever:
    def __init__(self,model_name:str = 'llama3',index_path:str='faiss_index'):
        self.model_name = model_name
        self.index_path = index_path
        try:
            
            self.embedder = OllamaEmbeddings(model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load Ollama embeddings: {e}")
        
    def retrieve(self,query:str,k:int=3):
        docs = FAISS.load_local(self.index_path,self.embedder,allow_dangerous_deserialization=True)
        return docs.similarity_search(query,k)
    


if __name__=='__main__':
    retriever = SaftyRetriever()
    query = "How can I prevent toddlers from electric shock?"
    results = retriever.retrieve(query,5)

    for i, doc in enumerate(results, 1):
        print(f"\n🔹 Result {i}: {doc.page_content}\nMetadata: {doc.metadata}")