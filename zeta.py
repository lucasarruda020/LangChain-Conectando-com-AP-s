## main_langgraph

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.runnables import RunnableConfig
import asyncio

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

class Rota(TypedDict):
    destino: Literal['praia', 'montanha']

def responder(pergunta :str):
    rota = assistente_roteador.invoke({"query":pergunta})
    if rota == "praia":
        return assistente_praia.invoke({"query": pergunta})
    return assistente_montanha.invoke({"query": pergunta})


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

class Estado(TypedDict):
    query:str
    destino: Rota
    resposta:str


async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await assistente_roteador.ainvoke({"query": estado["query"]}, config)}

async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await assistente_praia.ainvoke({"query": estado["query"]}, config)}

async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await assistente_montanha.ainvoke({"query": estado["query"]}, config)}


def escolher_no(estado:Estado)->Literal["praia", "montanha"]:
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

grafo = StateGraph(Estado)
grafo.add_node("roteador", no_roteador)
grafo.add_node("Praia", no_praia)
grafo.add_node("Montanha", no_montanha)


grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {"query": "quero visitar um lugar no Brasil famoso por praias e cultura"}
    )
    print(resposta["resposta"])

asyncio.run(main())