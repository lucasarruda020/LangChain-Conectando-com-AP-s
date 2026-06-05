from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

base_url = os.getenv("base_url")
api_key = os.getenv("api_key")

# Conecta ja direto com. chat sem precisar treinar por messages
modelo = ChatOpenAI(
    model='google/gemma-3-1b',
    base_url=base_url,
    api_key=api_key,
    temperature=0.5
)

## Test Modelo: resposta = modelo.invoke(prompt)
def contatollm(input_user):
    client_openai = OpenAI(
        base_url=base_url,
        api_key=api_key
    )


    resposta_do_llm = client_openai.chat.completions.create(
        model="google/gemma-3-1b",
        messages=[
            {"role": "system", 
             "content": """Voce eh uma IA de assistente de viagens."""},

            {"role": "user", 
             "content": f"{input_user}"}
        ],
        temperature=0.0
    )
    resposta = resposta_do_llm.choices[0].message.content.replace("```json", "").replace("```", "")
    print(resposta)
    return resposta


numero_dias = 7
numero_criancas = 2
atividade = "Praia"
prompt = f"Crie um roteiro de viagem de {numero_dias} dias, para uma familia com {numero_criancas} de criancas, que busca atividades como essas {atividade}"



modelo_de_prompt = PromptTemplate(
    """
    Crie um roteiro de viagem de {dias} dias, para uma familia com {numero_criancas} criancas, que gostam de {atividade}
    """
)

prompt = modelo_de_prompt.format(
    dias=numero_dias,
    numero_criancas = numero_criancas,
    atividade=atividade
)

