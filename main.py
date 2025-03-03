from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

set_debug(False)
set_verbose(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#loader = PyPDFLoader(file_path="./pdf/google_privacy_policy_en_us.pdf")
#docs = []
#docs_lazy = loader.lazy_load()
#for doc in docs_lazy:
#    docs.append(doc)


class DeepSeek():
    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large"):
        self.model = ChatOllama(model=llm_model)
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded privacy policy. 
            Your job is to answer questions about the privacy policy in a concise and accurate manner.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer concisely and accurately in three sentences or less.
            """
        )
        self.vector_store = None
        self.retriever = None


    def ingest(self, file_path: str):
        logger.info(f"Loading document from {file_path}")
        docs = PyPDFLoader(file_path=file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory="chroma_db"
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")
    

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)
    
    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None


if __name__ == "__main__":
    deepseek = DeepSeek()
    deepseek.ingest("./pdf/google_privacy_policy_en_us.pdf")
    query = "What information does Google collect?"
    response = deepseek.ask(query)
    print(response)
    deepseek.clear()