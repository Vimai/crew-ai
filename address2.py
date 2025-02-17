import json
import os
from typing import Optional

from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class AddressModel(BaseModel):
    address: str
    city: str
    state: str
    postal_code: str
    county: Optional[str] = None
    country: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


os.environ["OPENAI_API_KEY"] = "sk-proj-111"
os.environ["SERPER_API_KEY"] = ""

ollama_model = ChatOpenAI(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434"
)

serper_search_tool = SerperDevTool()

# Agente para analisar o endereço incompleto
analisador = Agent(
    role='Analisador de Endereço',
    goal='Identificar os dados do json e retornar o endereço como uma string',
    backstory='Um especialista em análise de endereços incompletos e identificação de informações ausentes.',
    tools=[],
    llm=ollama_model,
    verbose=True,
    max_iter=1
)

# Agente para preencher o endereço
preenchedor = Agent(
    role='Analista Senior de Endereço',
    goal='Completar os dados do endereços que estão ausentes e fornecer o endereço completo e preciso.',
    backstory='''Especialista em buscar na internet informaçoes sobre os endereços informados, para conseguir pegar todas as informaçoes referente ao endereco.
    Você sempre busca na internet com esse formato: 'Qual é o [campo faltante] para o meu endereco: [endereco que tenho]?' antes de buscar.
    ''',
    tools=[serper_search_tool],
    llm=ollama_model,
    verbose=True,
    max_iter=3
)

json_builder = Agent(
    role='Criador de json',
    goal='Cria json com os dados passados',
    backstory='Especialista em criar json.',
    llm=ollama_model,
    verbose=True,
    max_iter=1
)

analise_endereco = Task(
    description='Analisar o endereço e retornar os dados, informe quais dados estão faltando. Endereço: {endereco}',
    expected_output='Uma string com o endereço do json',
    agent=analisador
)

preenchimento_endereco = Task(
    description='Preencher os campos ausentes do endereço.Endereço: {endereco}',
    expected_output='Um JSON com o endereço completo.',
    agent=preenchedor
)


gerar_json = Task(
    description='Gera um json completo utilizando o json inicial e os dados recebidos do preenchedor de endereços. json inicial: {endereco}',
    expected_output='utilize o json inical e o resultado da task anteriror para criar o json com o endereço completo.',
    agent=json_builder,
    output_json=AddressModel
)

preenchimento_endereco.context = [analise_endereco]
gerar_json.context = [preenchimento_endereco]

# Equipe de agentes
equipe = Crew(
    agents=[analisador, preenchedor, json_builder],
    tasks=[analise_endereco, preenchimento_endereco, gerar_json],
    verbose=True
)

# Exemplo de entrada com endereço incompleto
endereco_incompleto = {
  "address": "4 Millner Avenue",
  "city": "Horsley Park",
  "state": "NSW",
  "postal_code": "2175",
  "county": "",
  "country": "Australia",
  "latitude": "",
  "longitude": ""
}


aaa = [
  {
    "address": "4 Millner Avenue",
    "city": "Horsley Park",
    "state": "NSW",
    "postal_code": "2175",
    "country": "Australia",
    "county": "",
    "latitude": "",
    "longitude": ""
  },
  {
    "address": "128, Direct Factory Outlet, Brisbane Airport",
    "city": "Brisbane",
    "state": "QLD",
    "postal_code": "7000",
    "country": "Australia",
    "county": "",
    "latitude": "",
    "longitude": ""
  },
  {
    "address": "SHOP 142 LEVEL 1,211 LATROBE STREET",
    "city": "MELBOURNE",
    "state": "",
    "postal_code": "3000",
    "country": "Australia",
    "county": "",
    "latitude": "",
    "longitude": ""
  },
  {
    "address": "Designer Outlet Straße 1, Unit 4120",
    "city": "Parndorf",
    "state": "",
    "postal_code": "7111",
    "country": "Austria",
    "county": "",
    "latitude": "",
    "longitude": ""
  },
    {
      "address": "",
      "city": "Heilbronn",
      "state": "Baden-Württemberg",
      "postal_code": "74081",
      "country": "Germany",
      "county": "",
      "latitude": "49.11865684",
      "longitude": "9.15510475"
    }
]

# {{'address': '123 Main Street', 'city': 'Anytown', 'state': 'CA', 'postal_code': '99999', 'county': None, 'country': 'null', 'latitude': None, 'longitude': None}
#   "address": "123 Main Street",
#   "city": "Anytown",
#   "state": "CA",
#   "postal_code": "91234",
#   "county": "Los Angeles",
#   "country": "USA",
#   "latitude": 34.0522,
#   "longitude": -118.2437
# }

# Executar as tarefas
result = equipe.kickoff({"endereco": json.dumps(aaa[2])})

print(result)
