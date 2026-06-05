## main_langgraph

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from typing import Literal, TypedDict
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

def responder(pergunta :str):
    rota = roteador.invoke({"query":pergunta})
    if rota == "praia":
        return assistente_praia.invoke({"query": pergunta})
    return assistente_montanha.invoke({"query": pergunta})

class Rota(TypedDict):
    destino: Literal['praia', 'montanha']

prompt_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr Praia, voce eh um especialista em viagens com destino para Verao + Praia"),
        ("human", "{query}")
    ]
)

prompt_montanhas = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr Montanha, voce eh um especialista em viagens com destino para escaladas"),
        ("human", "{query}")
    ]
)

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}")
    ]
)

assistente_praia = prompt_praia | modelo | StrOutputParser()
assistente_montanha = prompt_montanhas | modelo | StrOutputParser()
assistente_roteador = prompt_roteador | modelo.with_structured_output(Rota)


roteador = prompt_roteador
print(responder("Quero Surfar em um lugar quente"))