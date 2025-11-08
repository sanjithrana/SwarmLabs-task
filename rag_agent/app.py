# -------------------- IMPORTS --------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_community.llms import OpenAI
from langgraph.graph import StateGraph
import os
from langchain_groq import ChatGroq



# -------------------- SETUP --------------------
DATA_PATH = "rag_agent/data"
persist_directory = "chroma_store"

# Load text data
docs = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_PATH, file))
        docs.extend(loader.load())

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

# Create embeddings + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
#OPEN_AI_APIKEY = "sTqsWXrD3brY0rKagPlvJbOcy5zsUvEmRek4nwOFUMye2dCSoEITJ7tCEeJgSnooVWrTTqjhHp-p-Kr4J2StMzgUthmNk6rLLZPjb-hoUuHQCspLVyNyFJWvjoO6OEA"
#llm = OpenAI(model="gpt-3.5-turbo")
llm = ChatGroq(model="llama-3.1-8b-instant")
# -------------------- DEFINE STATE SCHEMA --------------------
class AgentState(TypedDict):
    query: str
    need_retrieval: Optional[bool]
    context: Optional[str]
    answer: Optional[str]
    reflection: Optional[str]

# -------------------- DEFINE NODES --------------------
def plan_node(state: AgentState) -> AgentState:
    print(" PLAN NODE: Deciding if retrieval is needed")
    query = state["query"]
    state["need_retrieval"] = any(q in query.lower() for q in ["what", "why", "how", "benefit", "explain"])
    return state

def retrieve_node(state: AgentState) -> AgentState:
    if not state.get("need_retrieval", True):
        print("ℹ️ Retrieval not required")
        return state
    print("🔍 RETRIEVE NODE: Fetching relevant documents")
    query = state["query"]
    docs = retriever.invoke(query)
    state["context"] = "\n".join([d.page_content for d in docs])
    return state

def answer_node(state: AgentState) -> AgentState:
    print(" ANSWER NODE: Generating answer with LLM")
    try:
        context = state.get("context", "")
        query = state["query"]
        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
        response = llm.invoke(prompt)
        state["answer"] = getattr(response, "content", str(response))

    except Exception as e:
        print(" LLM Error:", e)
        state["answer"] = "Sorry, I couldn't generate an answer due to an API issue."
    return state


def reflect_node(state: AgentState) -> AgentState:
    print("🪞 REFLECT NODE: Evaluating answer relevance")
    question = state["query"]
    answer = state["answer"]
    check_prompt = f"Is the following answer relevant to the question?\n\nQ: {question}\nA: {answer}\n\nAnswer YES or NO."
    reflection = llm.invoke(check_prompt)
    state["reflection"] = getattr(reflection, "content", str(reflection))

    return state

# -------------------- BUILD LANGGRAPH --------------------
graph = StateGraph(AgentState)  
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("reflect", reflect_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "reflect")

agent = graph.compile()

# -------------------- RUN --------------------
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    state: AgentState = {"query": user_query}
    result = agent.invoke(state)
    print("\n Final Answer:", result["answer"])
    print("Reflection:", result["reflection"])
