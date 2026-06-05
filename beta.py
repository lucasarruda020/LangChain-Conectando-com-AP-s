from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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
prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Voce eh um guia de viagem especializado, apresente-se como Sr Parceiros"),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "aula_langchain"

def historico_p_sessoa(sessao : str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura, pode sugerir?",
    "Qual a melhor epoca do ano para ir?"
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_p_sessoa,
    input_messages_key="query",
    history_messages_key="historico"

)

for pergunta in perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "query": pergunta,

        },
        config={"session_id": sessao}
    )
    print("Usuario: ", pergunta)
    print("IA: ", resposta, "\n")
    print("\n")



