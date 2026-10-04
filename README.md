# 🎬 CineData Analytics - Agente Text-to-SQL

Este projeto foi desenvolvido para a atividade de GenAI do Rocket Lab 2026 (Visagio). Ele consiste em um agente de IA capaz de converter perguntas em linguagem natural para consultas SQL, permitindo que usuários não técnicos analisem dados da camada Gold do catálogo de filmes (CineData).

## 🚀 Funcionalidades Implementadas
* **Interface Visual:** Desenvolvida em Streamlit para melhor experiência do usuário.
* **Motor Text-to-SQL:** Utilização da API do OpenRouter para tradução de queries.
* **Sistema de Fallback:** Alternância automática entre modelos gratuitos (`nvidia/nemotron-3.5-lightning:free`, `google/gemma-4-26b-a4b-it:free`, `openrouter/free`) caso o limite do provedor seja atingido.
* **Sistema de Cache:** Armazenamento local de perguntas repetidas (`cache.json`) para evitar o consumo desnecessário da cota diária (50 requisições/dia) do OpenRouter.
* **Visualização de Dados:** Geração automática de gráficos para tabelas numéricas.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Banco de Dados:** SQLite3 (`cinerocket.db`)
* **LLM Provider:** OpenRouter API
* **Bibliotecas:** `openai`, `pandas`, `streamlit`, `python-dotenv`

## ⚙️ Como Executar o Projeto

1. **Clone o repositório:**
   git clone https://github.com/RenanUS/cinedata-agent
   cd cinedata_agent

2. **Crie e ative um ambiente virtual:**
    python -m venv venv
    # No Windows:
    venv\Scripts\activate
    # No Linux/Mac:
    source venv/bin/activate

3. **Instale as dependências:**
    pip install -r requirements.txt
    
4. **Configure a Chave de API:**
    Crie um arquivo .env na raiz do projeto e adicione sua chave do OpenRouter:

    Snippet de código
    OPENROUTER_API_KEY=<SUA_CHAVE>
    (Certifique-se de que o arquivo cinerocket.db está presente na raiz da pasta).

5. **Execute a aplicação:**
    streamlit run app.py