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

question_list = {
    "Does the policy outline data collection practices?",
    "What is [the company]’s stance on government requests for user data?",
    "How does the policy address potential conflicts of interest in data usage or sharing?",
    "What sort of data is collected from me while using this?",
    "Are users given control over their data and privacy settings?",
    "Are there clear mechanisms for users to request data deletion or access?",
    "How does [the company] manage consent and withdrawal of consent from users?",
    "Can I opt out of letting them collect data and still use the app?",
    "Does [the company] minimize data retention periods?",
    "How is user data anonymized or aggregated to protect individual privacy?",
    "Are there any restrictions on data processing for specific purposes or contexts?",
    "How long is my data stored?",
    "Are user communications encrypted end-to-end?",
    "What measures are in place to prevent unauthorized access to user data?",
    "How are data breaches or security incidents handled and communicated to users?",
    "How well secured is my private information?",
    "Does [the company] conduct privacy impact assessments?",
    "Are there privacy-enhancing technologies implemented, such as differential privacy?",
    "Does [the company] use automated decision-making or profiling, and if so, how does it impact user privacy?",
    "What sort of analytics will my data be subjected to?",
    "Is the privacy policy regularly updated and communicated to users?",
    "Is there a process in place to address user privacy complaints?",
    "Does [the company] publish transparency reports detailing government data requests, surveillance, or law enforcement interactions?",
    "Has there ever been a security breach?",
    "Are employees trained on data privacy best practices and handling sensitive information?",
    "How are user data privacy preferences managed across different devices or platforms?",
    "Does [the company] offer user-friendly resources, such as tutorials or guides, to help users effectively manage their privacy settings and understand their data rights?",
    "Does it share any data with a third party?",
    "Does the policy comply with applicable privacy laws and regulations?",
    "What steps are taken to ensure data processors and subprocessors adhere to privacy requirements?"
    "Does [the company] have a process in place for reporting and addressing privacy violations or non-compliance issues, both internally and with third-party vendors?",
    "Do I have any rights as far as whether I want my account info deleted?"
}

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
        self.docs = None


    def ingest(self, file_path: str):
        logger.info(f"Loading document from {file_path}")
        self.docs = PyPDFLoader(file_path=file_path).load()
        """chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory="chroma_db"
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")"""
    

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        """if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."""

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in self.docs),
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

    """ f = open("answers.txt", "w")
    for query in question_list:
        response = deepseek.ask(query)
        f.write(f"Question: {query}\n")
        f.write(f"Answer: {response}\n")"""
    query = "Does the policy outline data collection practices?"
    response = deepseek.ask(query)
    print(response)
    deepseek.clear()