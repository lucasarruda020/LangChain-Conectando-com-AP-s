from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

load_dotenv()
base_url = os.getenv("base_url")
api_key_local = os.getenv("api_key")
api_key = os.getenv("OPENAI_API_KEY")


class Restaurante(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    restaurantes:str = Field("Motivo pelo qual restaurante eh recomendado")

parseador_restaurante = JsonOutputParser(pydantic_object=Restaurante)

class Destino(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    motivo:str = Field("motivo pelo qual é interessante visitar essa cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)


modelo = ChatOpenAI(
    model='google/gemma-3-1b',
    base_url=base_url,
    api_key=api_key_local,
    temperature=0.5
)

modelo_de_cidade = PromptTemplate(
    template="""
    sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

modelo_de_restaurantes = PromptTemplate(
    template="""
    sugira restaurantes populares entre locais em {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurante.get_format_instructions()}
)

modelo_de_cultura = PromptTemplate(
    template="""Sugira um passeio cultural em {cidade}"""
)

cadeia_cidade = modelo_de_cidade | modelo | parseador_destino
cadeia_restaurante = modelo_de_restaurantes | modelo | parseador_restaurante
cadeia_cultura = modelo_de_cultura | modelo | StrOutputParser()

cadeia = (cadeia_cidade | cadeia_restaurante | cadeia_cultura)

resposta = cadeia.invoke(
    {
        "interesse" : "praias"
    }
)

print(resposta)