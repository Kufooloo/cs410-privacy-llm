from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import os

set_debug(False)
set_verbose(False)



question_list = [
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
"What steps are taken to ensure data processors and subprocessors adhere to privacy requirements?",
"Does [the company] have a process in place for reporting and addressing privacy violations or non-compliance issues, both internally and with third-party vendors?",
"Do I have any rights as far as whether I want my account info deleted?",
]

paraphrased_questions = [
    "Does the policy specify how data is gathered?",
    "How does [the business] feel about requests for user data from the government?",
    "In what ways does the policy handle possible conflicts of interest when it comes to sharing or using data?",
    "What kind of information is gathered about me when I use this?",
    "Do people have authority over their privacy and data settings?",
    "Are there easy ways for users to seek access to or deletion of data?",
    "How does [the business] handle user consent and consent withdrawal?",
    "Can I use the app even if I choose not to allow them to gather my data?",
    "Does [the business] cut down on how long it keeps data?",
    "How is user data aggregated or anonymised to preserve personal privacy?",
    "Does the processing of data for certain contexts or purposes have any limitations?",
    "How long is my data kept on file?",
    "Do user communications have end-to-end encryption?",
    "What safeguards are in place to stop illegal access to user information?",
    "How are security incidents or data breaches handled and reported to users?",
    "To what extent is my personal information protected?",
    "Does [the business] carry out privacy impact analyses?",
    "Are technologies that improve privacy, such differential privacy, in use?",
    "What effects does [the company's] use of automated profiling or decision-making have on user privacy?",
    "What kind of analytics will be performed on my data?",
    "Does the privacy policy get updated and shared with users on a regular basis?",
    "Does a procedure exist for handling complaints about user privacy?",
    "Does [the business] release transparency reports that describe encounters with law enforcement, surveillance, or requests for government data?",
    "Has a security breach ever occurred?",
    "Do staff members receive training on managing sensitive data and best practices for data privacy?",
    "How are user choices for data privacy handled across various platforms or devices?",
    "Does [the business] provide easy-to-use tools, including tutorials or guides, to assist consumers in understanding their data rights and managing their privacy settings?",
    "Does it provide any third parties access to its data?",
    "Does the policy adhere to the rules and laws governing privacy?",
    "What measures are in place to guarantee that subprocessors and data processors follow privacy regulations?",
    "Does [the business] have a procedure in place for reporting and dealing with non-compliance concerns or privacy violations, both internally and with outside vendors?",
    "Do I have any rights regarding the deletion of my account information?"
]


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
            
            Answer concisely and accurately.
            Make sure to reference the portions of text that relate to the answer.
            """
        )
        self.vector_store = None
        self.retriever = None
        self.docs = None


    def ingest(self, file_path: str):
        #logger.info(f"Loading document from {file_path}")
        self.docs = PyPDFLoader(file_path=file_path).load()

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """

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

        #logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)
    
    def clear(self):
        """
        Reset the vector store and retriever.
        """
        #logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None



if __name__ == "__main__":
    deepseek = DeepSeek()
    pdfs = os.listdir("./pdf")
    count = 0
    print(len(paraphrased_questions))
    print(f"What pdf do you want to load?")
    for pdf in pdfs:
        print(f"{count}: {pdf}")
        count += 1
    select = int(input("Enter # of pdf here: "))


    
    deepseek.ingest(f"./pdf/{pdfs[select]}")
    mode = input("1. Ask Questions.\n2. Answer prewritten question suite and output result. \nSelect Mode:  ")
    query = ""
    while query != "exit":
        match mode: 
            case "1":
                while True:
                    query = input("Enter your question: ")
                    if query.lower() == "exit":
                        break
                    response = deepseek.ask(query)
                    print(response)
            case "2":
                count = 1
                f = open(f"answers_{pdfs[select]}.txt", "w", encoding='utf-8')
                for query in question_list:
                    response = deepseek.ask(query)
                    print(f"Question: {query}")
                    print(f"Answer: {response}")
                    f.write(f"Question {count}: {query}\n")
                    f.write(f"Answer: {response}\n")
                    count += 1
                f.close()
                count = 1
                f = open(f"answers_{pdfs[select]}_paraphrased.txt", "w", encoding='utf-8')
                for query in paraphrased_questions:
                    response = deepseek.ask(query)
                    print(f"Question: {query}")
                    print(f"Answer: {response}")
                    f.write(f"Question {count}: {query}\n")
                    f.write(f"Answer: {response}\n")
                    count += 1
                f.close()
                query = "exit"
                    
            case _:
                print("Invalid mode selected.")
                mode = input("1. Ask Questions.\n2. Answer prewritten question suite and output result. \nSelect Mode:  ")
