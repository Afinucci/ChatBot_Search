import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
import os
import re
from typing import List
from operator import itemgetter

# Set Streamlit to wide mode
st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Helper functions
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load or create FAISS index
def load_or_create_faiss_index():
    embeddings = HuggingFaceEmbeddings()

    if os.path.exists("faiss_index"):
        try:
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded from disk.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            db = None
    else:
        reference_documents_folder = "documents"
        loader = DirectoryLoader(reference_documents_folder, glob="*.pdf", loader_cls=PyMuPDFLoader)
        docs = loader.load()

        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(keep_separator=True, chunk_size=2000, chunk_overlap=0)
        splitted_docs = text_splitter.split_documents(docs)

        texts = [doc.page_content for doc in splitted_docs]

        db = FAISS.from_texts(texts, embeddings)
        db.save_local("faiss_index")

    return db

db = load_or_create_faiss_index()

# Define LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define classes
class CitedAnswer(BaseModel):
    answer: str
    citations: List[int]

class ChatSession:
    def __init__(self):
        self.history = []

    def add_interaction(self, user_input: str, cited_answer: CitedAnswer, docs: List[Document]):
        self.history.append({
            "user_input": user_input,
            "cited_answer": cited_answer,
            "docs": docs
        })

    def get_context(self) -> str:
        context = "\n".join([interaction["cited_answer"].answer for interaction in self.history])
        return context

    def chat(self, question: str):
        # Retrieve relevant documents before invoking the LLM
        docs = retrieve_documents(question)
        
        # Create context from previous history and the new documents
        formatted_docs = format_docs_with_id(docs)
        full_context = f"Previous Chat Context:\n{self.get_context()}\n\nUser Question: {question}\n\nRelevant Documents:\n{formatted_docs}"

        # Invoke the LLM using the full context and the relevant documents
        result = chain.invoke(full_context)
        cited_answer = result['cited_answer']

        # Add interaction to history
        self.add_interaction(question, cited_answer, docs)

        # Return answer, docs, and citations
        answer_output = self.format_answer(cited_answer, docs)
        return answer_output, docs, cited_answer.citations

    def format_answer(self, cited_answer, docs):
        answer_text = f"**Answer:**\n{cited_answer.answer}\n\n**Citations:**\n"
        for citation in cited_answer.citations:
            doc = docs[citation - 1]
            answer_text += f"- Document {citation}:\n"
            answer_text += f"  Title: {doc.metadata.get('title', 'No Title')}\n"
            answer_text += f"  Page: {doc.metadata.get('page', 'Unknown Page')}\n"
        return answer_text

# Retrieve documents using FAISS
def retrieve_documents(question: str) -> List[Document]:
    return db.similarity_search(question, k=6)

# Format retrieved documents for the LLM
def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Document {i+1}:\nTitle: {doc.metadata.get('title', 'No Title')}\nPage: {doc.metadata.get('page', 'Unknown Page')}\nContent Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n".join(formatted)

# Cited answer extraction tool
def cited_answer_tool(context: str) -> CitedAnswer:
    response = llm.invoke(f"Context: {context}\nAnswer based only on the given sources, and cite them using Document numbers.")

    response_content = response.content

    # Extract citations using regex
    if "Citations:" in response_content:
        answer = response_content.split("Citations:")[0].strip()
        citations_raw = response_content.split("Citations:")[1].strip()

        # Extract citation numbers from Document references
        citations = [int(re.search(r"Document (\d+)", x).group(1)) for x in re.findall(r'Document \d+', citations_raw)]
    else:
        answer = response_content.strip()
        citations = []

    return CitedAnswer(answer=answer, citations=citations)

# Define chain using FAISS document retrieval and LLM answer generation
cited_answer_runnable = RunnableLambda(cited_answer_tool)

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=RunnableLambda(retrieve_documents))
    .assign(context=itemgetter("docs") | RunnableLambda(format_docs_with_id))
    .assign(cited_answer=cited_answer_runnable)
    .pick(["cited_answer", "docs"])
)

# Streamlit app layout
st.title("Chatbot Based Search")

# Initialize the chat session
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = ChatSession()

# User input
user_input = st.text_input("Ask a question:", "")

# Layout with two columns
col1, col2 = st.columns([1, 1])

if user_input:
    try:
        # Get the answer, documents, and citations
        answer, docs, citations = st.session_state.chat_session.chat(user_input)
        
        # Display the chat history and answer on the left column
        with col1:
            st.subheader("Chat History")
            for interaction in reversed(st.session_state.chat_session.history):
                st.markdown(f"**You:** {interaction['user_input']}")
                st.markdown(f"**Assistant:** {interaction['cited_answer'].answer}")
        
        # Display referenced documents in the right column
        with col2:
            st.subheader("Referenced Documents")
            if docs:
                for i, doc in enumerate(docs):
                    title = doc.metadata.get('title') or doc.metadata.get('source', f"Document {i+1}")
                    st.markdown(f"**Document {i+1}:**")
                    st.markdown(f"Title: {title}")
                    st.markdown(f"Page: {doc.metadata.get('page', 'Unknown Page')}")
                    st.markdown(f"Content: {doc.page_content}")
            else:
                st.markdown("No documents retrieved.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
