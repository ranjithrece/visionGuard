import json
import os
from langchain_ollama import OllamaEmbeddings
from langchain.schema import  Document
from langchain_community.vectorstores import FAISS
from typing import List
 
class IndexBuilder:
    def __init__(self, model_name: str = "llama3", index_path: str = "faiss_index"):
        """
        Initializes the embedding model and index path.

        Args:
            model_name (str): The name of the local Ollama model to use.
            index_path (str): Directory path where FAISS index will be saved.
        """
        self.model_name = model_name
        self.index_path = index_path

        try:
            self.embeddings = OllamaEmbeddings(model=self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load Ollama embeddings: {e}")

   #load json into langchain Document type
    def load_documents(self, file_path: str) -> List[Document]:
        
        documents = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                doc = Document(page_content=data['text'], metadata=data['metadata'])
                documents.append(doc)
        return documents

    #build faiss index 
    def build_index(self, documents: List[Document]):
            db = FAISS.from_documents(documents,self.embeddings)
            db.save_local(self.index_path)
            print("✅ FAISS index saved successfully.")

    #run function
    def run(self, file_path: str):
        print(f"Loading documents from: {file_path}")
        documents = self.load_documents(file_path)
        print(f"Loaded {len(documents)} documents. Building index...")
        self.build_index(documents)
        

if __name__ == "__main__":
    builder = IndexBuilder(model_name="llama3", index_path="faiss_index")
    builder.run("vector_store/data/safety_knowledge.jsonl")
