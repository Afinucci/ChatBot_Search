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
import pprint
from operator import itemgetter


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Define helper functions
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load or create FAISS index
def load_or_create_faiss_index():
    embeddings = HuggingFaceEmbeddings()

    # Check if FAISS index exists
    if os.path.exists("faiss_index"):
        try:
            # Load the existing FAISS index from the disk
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            print("#################FAISS index loaded successfully from disk.######################")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            db = None
    else:
        print("####################FAISS index not found. Creating a new one...########################3")

        # Load documents from the directory
        reference_documents_folder = "documents"
        loader = DirectoryLoader(reference_documents_folder, glob="*.pdf", loader_cls=PyMuPDFLoader)
        docs = loader.load()

        # Clean and split the documents into chunks
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(keep_separator=True, chunk_size=1000, chunk_overlap=200)
        splitted_docs = text_splitter.split_documents(docs)

        # Extract the text content from each document for embedding
        texts = [doc.page_content for doc in splitted_docs]

        # Create FAISS index from the text content
        db = FAISS.from_texts(texts, embeddings)
        
        # Save the FAISS index locally
        db.save_local("faiss_index")
        print("####################3FAISS index created and saved successfully.##############################")

    return db

# Load the FAISS index or create it if it doesn't exist
db = load_or_create_faiss_index()

"""
#Use the following lines for Testing
query = "what is an Isolator?"
queried_docs = db.similarity_search(query, k = 6)

i=1
for chunk in queried_docs:
  print(f"Chunk: {i}\n")
  pprint.pprint(chunk.page_content)
  print("#" *100)
  i+=1
#Testing end
"""

#Defining the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)

# Define the CitedAnswer and ChatSession classes as provided
class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""
    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources."
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer."
    )

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
        # Combine all previous answers to create context
        context = "\n".join([interaction["cited_answer"].answer for interaction in self.history])
        return context

    def chat(self, question: str):
        # Get the previous context
        context = self.get_context()

        # Combine the context with the new question
        full_context = f"{context}\n\nUser Question: {question}"

        # Process the chain with the new question
        result = chain.invoke(full_context)

        # Extract the cited answer and documents
        cited_answer = result['cited_answer']
        docs = result['docs']

        # Map citations to document references
        citation_map = {i + 1: f"Document {i + 1}" for i in range(len(docs))}

        # Update the history with this interaction
        self.add_interaction(question, cited_answer, docs)

        # Format the output
        answer_output = self.format_answer(cited_answer, docs, citation_map)
        return answer_output

    def format_answer(self, cited_answer, docs, citation_map):
        answer_text = f"**Answer:**\n{cited_answer.answer}\n\n**Citations:**\n"
        if cited_answer.citations:
            for citation in cited_answer.citations:
                doc = docs[citation - 1]
                answer_text += f"- {citation_map[citation]}:\n"
                answer_text += f"  Title: {doc.metadata.get('title', 'No Title')}\n"
                answer_text += f"  Source: {doc.metadata.get('source', 'No Source')}\n"
                answer_text += f"  Page: {doc.metadata.get('page', 'Unknown Page')}\n"
        else:
            answer_text += "No citations available."

        documents_text = "\n**Documents:**\n"
        for i, doc in enumerate(docs):
            documents_text += f"\nDocument {i+1}:\n"
            documents_text += f"Title: {doc.metadata.get('title', 'No Title')}\n"
            documents_text += f"Source: {doc.metadata.get('source', 'No Source')}\n"
            documents_text += f"Page: {doc.metadata.get('page', 'Unknown Page')}\n"
            documents_text += "Content Snippet:\n"
            documents_text += doc.page_content + "...\n"  # Limiting to 500 characters for brevity

        return f"{answer_text}\n\n{documents_text}"

# Function to format documents with their IDs
def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Document {i+1}:\nTitle: {doc.metadata.get('title', 'No Title')}\nSource: {doc.metadata.get('source', 'No Source')}\nPage: {doc.metadata.get('page', 'Unknown Page')}\nContent Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)

# Create a function to retrieve documents using FAISS
def retrieve_documents(question: str) -> List[Document]:
    return db.similarity_search(question, k=6)

# Create the necessary Runnables
format = itemgetter("docs") | RunnableLambda(format_docs_with_id)

# Modified cited_answer_tool function
def cited_answer_tool(context: str) -> CitedAnswer:
    response = llm.invoke(f"Context: {context}\nAnswer based only on the given sources, and cite them using Document numbers (e.g., Document 1, Document 2).")

    response_content = response.content

    if "Citations:" in response_content:
        answer = response_content.split("Citations:")[0].strip()
        citations_raw = response_content.split("Citations:")[1].strip()

        # Extract document numbers from the citations
        citations = [int(x.split()[-1]) for x in citations_raw.split(",") if "Document" in x]
    else:
        answer = response_content.strip()
        citations = []

    return CitedAnswer(answer=answer, citations=citations)

cited_answer_runnable = RunnableLambda(cited_answer_tool)

# Adjust the input to be a simple string (the question)
chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=RunnableLambda(retrieve_documents))
    .assign(context=format)
    .assign(cited_answer=cited_answer_runnable)
    .pick(["cited_answer", "docs"])
)



def interactive_chat():

    # Initialize the ChatSession
    chat_session = ChatSession()
    
    while True:
        # Prompt the user for input
        user_input = input("User: ")
        
        # Check for the exit condition
        if user_input.strip().lower() == 'exit':
            print("Exiting chat session.")
            break
        
        # Use chat_session.chat() to submit the question and get the assistant's reply
        try:
            assistant_reply = chat_session.chat(user_input)
            print(f"Assistant: {assistant_reply}\n")
        except Exception as e:
            print(f"An error occurred: {e}")
            break

# Start the interactive chat session
interactive_chat()