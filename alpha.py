from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
base_url = os.getenv("base_url")
api_key_local = os.getenv("api_key")
api_key = os.getenv("OPENAI_API_KEY")


numero_dias = 7
numero_criancas = 2
atividade = "Praia"
prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma familia com {numero_criancas} de criancas, que busca atividades como essas {atividade}"


modelo = ChatOpenAI(
    model='google/gemma-3-1b',
    base_url=base_url,
    api_key=api_key_local,
    temperature=0.5
)
modelo_de_cidade = PromptTemplate(
    template="""
    sugira uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=["interesse"]
)

cadeia = modelo_de_cidade | modelo | StrOutputParser()

resposta = cadeia.invoke(
    {
        "interesse": "praias"
    }
)

print(resposta)