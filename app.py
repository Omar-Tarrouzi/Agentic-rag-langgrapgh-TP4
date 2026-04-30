from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage  # Correction: langchain.messages → langchain_core.messages
from langchain.tools import tool  # AJOUT OBLIGATOIRE: import du décorateur @tool

load_dotenv(override=True)

loader = PyPDFLoader("CV 4.pdf")
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=50
)

chunks = loader.load_and_split(splitter)
embedding_model = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="cv_data_collection",
    persist_directory="cv_data_collection"  # Correction: ajout de la virgule manquante
)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

@tool
def retriever_tool(query: str) -> str:
    """
    permet de chercher des infos sur des candidats
    comme nom, prenom et diplome dans cv 4
    """
    relevant_documents = retriever.invoke(query)  # Correction: relevent → relevant
    context_list = [d.page_content for d in relevant_documents]
    context = " ".join(context_list)  # Correction: "." → " " (espace plutôt que point)
    return context

@tool
def get_company_info(companyname: str):
    """
    consulter des infos sur l'entreprise donnée
    """
    return {
        "companyname": companyname,
        "domain": "IT",
        "turnover": 120_870_000
    }

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent2 = create_agent(
    model=llm,
    tools=[retriever_tool, get_company_info],
    system_prompt="Répond à la question de l'utilisateur avec les tools fournis"  # Correction: fournit → fournis
)