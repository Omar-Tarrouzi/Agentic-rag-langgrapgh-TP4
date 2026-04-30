import warnings
warnings.filterwarnings("ignore")

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import BaseTool
from typing import Type, Optional, List
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(override=True)

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

texts = [
    "Je m'appelle Omar Tarrouzi en Informatique et Intelligence artificielle",
    "J'étudie à l'emsi de Casablanca",
    "i'm more than meet the eye",
]

vectorstore = Chroma.from_texts(texts, embedding_model, collection_name="Agentic_AI")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ CRÉATION MANUELLE DE L'OUTIL RETRIEVER (fonctionne toujours)
class RetrieverInput(BaseModel):
    query: str = Field(description="The search query to find information")

class CustomRetrieverTool(BaseTool):
    name: str = "kb_search"
    description: str = "Search information about me"
    retriever: any = None
    
    def _run(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if not docs:
            return "No information found."
        return "\n\n".join([doc.page_content for doc in docs])
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

retrieval_tool = CustomRetrieverTool(retriever=retriever)

# send_mail tool
@tool
def send_mail(email: str, subject: str, content: str):
    """Send email to the given email with the provided subject and content"""
    print("=="*50)
    print("send_mail tool invoked")
    print("=="*50)
    return f"this email has been sent : destination : {email}, subject: {subject}, content: {content}"

# get_employee_info tool
@tool
def get_employee_info(name: str):
    """Get info about employee (name, salary, seniority)"""
    print("=="*50)
    print("get_employee_info tool invoked")
    print("=="*50)
    return {"name": name, "salary": 5000, "seniority": 5}

# create agent
graph = create_agent(
    model=llm,
    tools=[get_employee_info, retrieval_tool, send_mail],
    system_prompt="answer the user question using provided tools",
)

# invoke agent
resp = graph.invoke(
    input={"messages": [HumanMessage("Je veux connaitre le salaire de Yassine")]}
)
print(resp["messages"][-1].content)