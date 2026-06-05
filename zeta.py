## main_langgraph

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

base_url = os.getenv("base_url")
api_key_local = os.getenv("api_key")
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model='google/gemma-3-1b',
    base_url=base_url,
    api_key=api_key_local,
    temperature=0.5
)

prompt_consultor = ChatPromptTemplate.from_messages(
    [
        ("system", "voce eh um consultor de viagens"),
        ("human", "{query}")
    ]
)


assistente = prompt_consultor | modelo | StrOutputParser()

resp = assistente.invoke({"query": "Quero ferias em praias do Brasil"})
print(resp)