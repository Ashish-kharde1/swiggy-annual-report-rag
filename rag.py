import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
import os
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Configuration
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")
MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def format_docs(docs):
    return "\n\n".join(
        f"(Page {d.metadata.get('page', 'N/A')}) {d.page_content}"
        for d in docs
    )

def get_rag_chain():
    """
    Initializes and returns a LangChain 1.x compatible RAG chain.
    """

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. Run ingest.py first."
        )

    # 1️⃣ Load embeddings + vector store
    embeddings = EMBEDDING_MODEL

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.5,
    },
)


    # 2️⃣ LLM
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
    )

    # 3️⃣ Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant answering questions ONLY using the provided context from the Swiggy Annual Report.

Rules:
- Use ONLY the information present in the context.
- You may summarize information ONLY if the context explicitly refers to the same metric and scope.
- Do NOT substitute related but different metrics (e.g., Food Delivery orders vs total B2C orders).
- Do NOT use prior knowledge.
- Do NOT make assumptions.
- If the exact metric or phrasing asked in the question is not present in the context,
respond ONLY with:
"I could not find this information in the Swiggy Annual Report based on the retrieved context."
-Do NOT provide partial or related values.
- Keep answers concise, factual, and grounded.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def retrieve_and_format(question):
        docs = retriever.invoke(question)
        return {
            "context": format_docs(docs),
            "question": question,
            "source_documents": docs,
        }

    rag_chain = (
        RunnableLambda(retrieve_and_format)
        | {
            "answer": prompt | llm | StrOutputParser(),
            "source_documents": RunnableLambda(lambda x: x["source_documents"]),
        }
    )

    return rag_chain