from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from vector_store.searcher import SaftyRetriever
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json


# 1. Define the state structure passed between LangGraph nodes
class SafetySearchState(TypedDict):
    query: str
    results: List[Document]
    summary: str
    img_path: str
    caption: str
    evaluation: str
    evaluation_reason: str


# 2. LangGraph pipeline class for VisionGuard
class VisionGuardGraph:
    def __init__(self):
        """
        Initializes the SaftyRetriever and sets up graph nodes.
        """
        self.retriever = SaftyRetriever()
        self.llm = OllamaLLM(model="llama3")
        self.vision_llm = OllamaLLM(model="llava")

    def caption_node(self, state:SafetySearchState) -> SafetySearchState:
        """
        Read image from source and generate caption"""
        path = state["img_path"]
        prompt = """You are a visual safety assistant analyzing real-world hazards in images.

        Look at the image and return a concise, clear safety-focused description. Your response should include:

        1. Who or what is involved (e.g., child, object, tool)
        2. The visible hazard or dangerous situation
        3. The specific risk (e.g., electrocution, injury, fire)
        Avoid vague or emotional language. Use terms that are relevant to safety inspections and training.

        Respond in **2–3 sentences**.
        """
        caption = self.vision_llm.invoke(prompt,images=[path])
        state["caption"] = caption
        response = self.caption_summarizer(caption)
        state["query"] = response
        return state
    def caption_summarizer(self, query) -> SafetySearchState:
        prompt = PromptTemplate.from_template("""You are a retrieval engineer helping an AI safety assistant.

        Given this detailed image caption:

        "{caption}"

        Generate a short and focused search query (not a question) to retrieve safety guidelines, best practices, and hazard risks related to the situation. Avoid fluff or full sentences.

        The query should include:
        - The subject (e.g. child, outlet)
        - The specific hazard (e.g. electrocution)
        - Any keywords that relate to prevention (e.g. childproofing, outlet cover)
        only summary DO NOT Explain or restate
        """)
        format_prompt = prompt.format(caption=query)
        return self.llm.invoke(format_prompt)


    def search_node(self, state: SafetySearchState) -> SafetySearchState:
        """
        LangGraph node that performs vector search.
        """
        query = state["query"]
        results = self.retriever.retrieve(query)
        state["results"] = results
        return state

    def summarize_node(self,state:SafetySearchState) -> SafetySearchState:
        """
        Summarizing the retrieval results"""
        prompt = PromptTemplate.from_template(
                    "Given the question:\n{question}\n\nSummarize the following safety information into clear advice:\n{context}"

        )
        docs = state["results"]
        context = "\n\n".join(doc.page_content for doc in docs)
        formatPrompt = prompt.format(question=state["query"],context=context)

        response = self.llm.invoke(formatPrompt)
        state['summary'] = response
        return state
    
    def evaluate_node(self,state:SafetySearchState) -> SafetySearchState:
        """
        Evaluator node"""
        prompt = PromptTemplate.from_template("""You are an evaluator for a safety QA assistant.

        Given:
        - Query: {query}
        - Retrieved documents: {context}
        - Final summary: {summary}

        Evaluate if the summary clearly and completely answers the query using the retrieved information.

        Respond with:
         a only one list of 2 elements                                     
        1.in a scale of 0 to 1 based on the relevence to the question.  2.A brief explain of why you gave that rating. eg[1,"it is perfect cause all esponse is related]
        GIVE ONLY THE LIST ,NO OTHER EXTRA
                
        """)
        docs = state["results"]
        context = "\n\n".join(doc.page_content for doc in docs)
        formated_prompt = prompt.format(query=state["query"],context = context,summary=state['summary'])
        response = self.llm.invoke(formated_prompt)
        state['evaluation'],state["evaluation_reason"] = json.loads(response)

         
        return state
    
    
    def build(self):
        """
        Builds and compiles the LangGraph graph for safety search.
        """
        graph = StateGraph(SafetySearchState)
        graph.add_node("caption", RunnableLambda(self.caption_node))
        graph.add_node("search", RunnableLambda(self.search_node))
        graph.add_node("summary",RunnableLambda(self.summarize_node))
        graph.add_node("evaluation", RunnableLambda(self.evaluate_node))
        graph.add_edge("caption","search")
        graph.add_edge("search", "summary")
        graph.add_edge("summary","evaluation")
        graph.set_entry_point("caption")
        graph.set_finish_point("summary")
        return graph.compile()






# 3. Run the graph for a test query
if __name__ == "__main__":
    graph_builder = VisionGuardGraph()
    graph = graph_builder.build()

    input_state = {
    "img_path": "test_images/kid_near_socket.webp",
    "query": "",
    "results": [],
    "summary": ""
}


    output = graph.invoke(input_state)
    print("📸 Caption:", output["caption"])
    print("🔎 Query:", output["query"])
    print("📄 Results:", [d.page_content[:100] for d in output["results"]])
    print("🧠 Summary:", output["summary"])

