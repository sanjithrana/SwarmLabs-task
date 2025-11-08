# SwarmLabs-task
🧠 AI Q&A Agent using LangGraph + RAG

A fully functional Retrieval-Augmented Generation (RAG) agent built using LangGraph and LangChain, capable of answering user questions from a local knowledge base.

This project demonstrates a complete AI agent workflow — planning, retrieval, answer generation, and reflection — with support for OpenAI, Groq, or Hugging Face models.

📋 Features

✅ End-to-End RAG Pipeline — Retrieve relevant data and generate contextual answers.
✅ LangGraph Workflow — Modular AI agent with plan → retrieve → answer → reflect nodes.
✅ Multiple LLM Support — Works with OpenAI, Groq, or Hugging Face models.
✅ Vector Database (Chroma) — Stores embeddings for fast document retrieval.
✅ Fallback Mode — Automatically switches to a free Hugging Face model if API keys fail.
✅ Streamlit UI (optional) — Interact with the agent using a simple web app.
✅ Evaluation Module — Compute BLEU/ROUGE scores or judge with an LLM.

🏗️ Project Structure
rag_agent/
│
├── app.py                 # Main LangGraph + RAG agent
├── ui.py                  # (Optional) Streamlit interactive UI
├── evaluate_agent.py      # Evaluation module (ROUGE, BLEU, LLM Judge)
├── requirements.txt       # All dependencies
├── README.md              # This file
└── data/                  # Knowledge base folder
    ├── renewable_energy.txt
    ├── artificial_intelligence.txt
    ├── machine_learning.txt
    ├── data_science.txt
    ├── ethics_in_ai.txt
    ├── cloud_computing.txt
    └── future_of_ai.txt

⚙️ Setup Instructions
1️⃣ Clone or Download the Repository
git clone https://github.com/yourusername/rag_agent.git
cd rag_agent

2️⃣ Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate  # (Windows)
source venv/bin/activate  # (Mac/Linux)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add API Keys

Create a .env file inside rag_agent/ and add your keys:

OPENAI_API_KEY=sk-your_openai_key_here
GROQ_API_KEY=gsk-your_groq_key_here
HUGGINGFACEHUB_API_TOKEN=hf_your_huggingface_token_here


If OpenAI or Groq keys fail, the app automatically switches to Hugging Face.

🚀 Running the Project
▶️ Run the RAG Agent (CLI)
python app.py


Then type your question, for example:

What are the benefits of renewable energy?


Example Output:

🧩 PLAN NODE: Deciding if retrieval is needed
🔍 RETRIEVE NODE: Fetching relevant documents
💬 ANSWER NODE: Generating answer with LLM
🪞 REFLECT NODE: Evaluating answer relevance

✅ Final Answer:
According to the context, the benefits of renewable energy include:
1. Sustainability
2. Energy independence
3. Reduced pollution
4. Job creation

Reflection: YES

💻 (Optional) Run Streamlit UI
streamlit run ui.py


This opens an interactive Q&A web interface.

🧾 Evaluation (Optional)

Run the evaluation script to measure RAG quality:

python evaluate_agent.py


It computes:

ROUGE Score

BLEU Score

LLM-as-a-Judge (optional GPT evaluation)

🧩 Key Components
Node	Function
Plan	Understands user intent and decides if retrieval is needed.
Retrieve	Fetches relevant documents from Chroma vector store.
Answer	Generates a contextual answer using the LLM.
Reflect	Evaluates answer completeness and relevance.
📚 Technologies Used
Category	Tools
Framework	LangGraph, LangChain
Vector DB	ChromaDB
Embeddings	Hugging Face (sentence-transformers/all-MiniLM-L6-v2)
LLMs	OpenAI / Groq / Hugging Face
Frontend (optional)	Streamlit
Evaluation	ROUGE, BLEU, LLM-as-Judge
🧠 Example Knowledge Base Topics

Renewable Energy

Artificial Intelligence

Machine Learning

Data Science

Cloud Computing

AI Ethics

Future of AI

🛠️ Troubleshooting
Issue	Cause	Solution
Invalid API Key	Wrong or missing .env key	Update .env with valid API key
RateLimitError	Free API quota used up	Switch to Groq or HuggingFace model
FileNotFoundError: data	Missing folder	Create /data/ and add .txt files
_thread.RLock warning	Windows multiprocessing issue	Run pip install -U multiprocess dill
JSON output from LLM	LLM returned object	Fixed via .content extraction in code
🧰 Requirements

See requirements.txt for the complete list:

langchain
langchain-core
langchain-community
langchain-openai
langchain-huggingface
langchain-text-splitters
langgraph
chromadb
sentence-transformers
tiktoken
openai
python-dotenv
streamlit
evaluate
transformers
multiprocess
dill


Install with:

pip install -r requirements.txt

💡 Future Enhancements

Add memory for multi-turn conversations.

Integrate LangSmith or TruLens for trace logging.

Add web-based document upload for custom RAG context.

Use local models (Llama 3, Mistral) for offline usage.

🧑‍💻 Author

Developed by: sanjith
Contact: your.chilupurisanjith18@gmail.com


GitHub: github.com/yourusername
