import streamlit as st
from app import agent

st.set_page_config(page_title="LangGraph RAG Agent", layout="centered")
st.title("AI Q&A Agent (LangGraph + RAG)")

st.markdown("Ask any question from your knowledge base below ")

query = st.text_input("Enter your question:")
if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking..."):
            result = agent.invoke({"query": query})
        st.success("Answer generated successfully!")
        st.markdown(f"###**Answer:** {result['answer']}")
        st.markdown(f"**Reflection:** {result['reflection']}")
