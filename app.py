import sqlite3
import os
import re
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Carrega a variável OPENROUTER_API_KEY do arquivo .env
load_dotenv()

# Configuração da API do OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Lista de modelos para Fallback (do mais rápido/adequado para SQL para os gerais)
MODELOS = [
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free"
]

CACHE_FILE = "cache.json"
DB_FILE = "cinerocket.db"
MAX_BARRAS = 25

# Perguntas de exemplo (agrupadas nas categorias do enunciado)
EXEMPLOS = {
    "💰 Bilheteria e Finanças": [
        "Top 10 filmes com maior receita",
        "Lucro médio por gênero, considerando apenas filmes com receita informada",
        "Filmes com maior margem de lucro, entre os que possuem receita e orçamento informados",
    ],
    "🔥 Popularidade e Engajamento": [
        "Os 5 filmes mais populares",
        "Filmes com maior divergência entre a nota TMDB e a nota IMDb",
        "Nota média IMDb por ano de lançamento",
    ],
    "🎭 Elenco e Equipe": [
        "Ator com mais participações em filmes lançados nos últimos 5 anos",
        "Diretores com maior nota média (mínimo de 5 filmes)",
        "Dupla ator–diretor que mais trabalhou junta",
    ],
    "🎞️ Gêneros e Produtoras": [
        "Quantidade de filmes por gênero",
        "Produtora com maior lucro total",
        "Gênero com maior margem de lucro média",
    ],
    "⭐ Avaliações dos Usuários": [
        "Filmes mais avaliados pelos usuários",
        "Filmes em que a nota média dos usuários mais diverge da nota IMDb",
    ],
}

# --- FUNÇÕES DE CACHE ---
def carregar_cache():
    """Carrega o histórico de perguntas para não gastar a cota da API."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_cache(cache_dict):
    """Salva a nova pergunta e query SQL no arquivo."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_dict, f, ensure_ascii=False, indent=4)

# --- FUNÇÕES DE BANCO DE DADOS ---
def conectar():
    """Abre o banco em modo somente leitura (não cria arquivo vazio se ele não existir)."""
    return sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)

@st.cache_data
def obter_esquema():
    """Extrai o esquema de criação (DDL) das 10 tabelas do banco cinerocket.db"""
    try:
        conn = conectar()
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        esquemas = cursor.fetchall()
        conn.close()
        return "\n".join([esq[0] for esq in esquemas if esq[0]])
    except Exception as e:
        return f"Erro ao ler banco: {e}"

def executar_sql(query):
    """Executa a query e retorna um DataFrame do Pandas para facilitar a exibição visual."""
    try:
        conn = conectar()
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

# --- GUARDRAIL ---
PALAVRAS_PROIBIDAS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|TRUNCATE)\b",
    re.IGNORECASE,
)

def validar_sql(sql):
    """Permite apenas uma consulta de leitura (SELECT / WITH). Retorna (ok, sql_limpo, motivo)."""
    limpo = sql.strip().rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", limpo, re.IGNORECASE):
        return False, limpo, "A consulta gerada não é uma leitura (SELECT)."
    if ";" in limpo:
        return False, limpo, "Múltiplos comandos SQL não são permitidos."
    if PALAVRAS_PROIBIDAS.search(limpo):
        return False, limpo, "A consulta contém comandos de escrita ou administração."
    return True, limpo, None

# --- FUNÇÃO DO AGENTE (LLM) ---
def gerar_sql(pergunta, esquema):
    """Usa o OpenRouter para gerar SQL com sistema de Fallback. Retorna (sql, info, veio_do_cache)."""
    chave = pergunta.strip().lower()
    cache = carregar_cache()
    if chave in cache:
        return cache[chave], "Retornado do cache ⚡", True

    prompt = f"""
    Você é um Engenheiro de Dados especialista em SQLite.
    Sua missão é traduzir a pergunta do usuário para uma query SQL baseada no esquema abaixo.

    REGRAS DE NEGÓCIO:
    - "Receita", "Faturamento" e "Bilheteria" significam a mesma coisa.
    - Se a pergunta envolver "Lucro", calcule: (revenue - budget).
    - Se pedir listas (ex: "Top 10", "Os mais populares"), não esqueça do ORDER BY e LIMIT.
    - Gere apenas consultas de leitura (SELECT).

    ESQUEMA DO BANCO:
    {esquema}

    Retorne EXCLUSIVAMENTE o código SQL, sem crases (```) e sem explicações textuais.
    """

    for modelo in MODELOS:
        try:
            response = client.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": pergunta}
                ]
            )
            sql = response.choices[0].message.content.strip()
            sql = sql.replace("```sql", "").replace("```", "").strip()
            return sql, f"Gerado pelo modelo: {modelo}", False

        except Exception as e:
            # Se for erro 429 ou lotado, o laço continua para o próximo modelo (Fallback)
            print(f"Erro no modelo {modelo}: {e}")
            continue

    return None, "Todos os modelos gratuitos falharam (limite atingido ou provedores lotados).", False

# --- VISUALIZAÇÃO ---
def colunas_numericas(df):
    return [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]

def eixo_x_valido(df):
    """O eixo X deve ser um rótulo (texto) ou um ano."""
    x = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[x]):
        return True
    return ehano(df, x)

def ehano(df, col):
    nome = col.lower()
    if "ano" in nome or "year" in nome:
        return True
    serie = df[col].dropna()
    return not serie.empty and serie.between(1880, 2100).all() and (serie % 1 == 0).all()

def desenhar_grafico(df, y):
    """Escolhe o tipo de gráfico pelo formato do resultado: linha para anos, barras horizontais para rankings."""
    x = df.columns[0]
    if pd.api.types.is_numeric_dtype(df[x]):
        fig = px.line(df.sort_values(x), x=x, y=y, markers=True)
        fig.update_traces(line_color="#E50914")
        fig.update_xaxes(type="category" if df[x].nunique() < 15 else "linear")
    else:
        dados = df.head(MAX_BARRAS)
        fig = px.bar(
            dados, x=y, y=x, orientation="h", text_auto=".3s",
            color=y, color_continuous_scale="Reds",
        )
        # mantém a ordem do SQL (maior no topo)
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=max(320, 28 * min(len(df), MAX_BARRAS) + 80) if not pd.api.types.is_numeric_dtype(df[x]) else 400,
    )
    return fig

def configurar_colunas(df):
    """Formata moeda e notas automaticamente pelo nome da coluna."""
    cfg = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        nome = col.lower()
        serie = df[col].dropna()
        if any(k in nome for k in ("revenue", "receita", "budget", "orcamento", "orçamento",
                                    "lucro", "profit", "faturamento", "bilheteria")) \
                and not any(k in nome for k in ("marg", "perc", "pct")):
            cfg[col] = st.column_config.NumberColumn(col, format="R$ %.0f")
        elif any(k in nome for k in ("nota", "rating", "vote_average", "imdb", "tmdb")) \
                and not any(k in nome for k in ("dif", "diverg", "delta")) \
                and not serie.empty and serie.between(0, 10).all():
            cfg[col] = st.column_config.ProgressColumn(col, min_value=0, max_value=10, format="%.1f")
    return cfg

def formatar_metrica(valor):
    if isinstance(valor, (int,)) or (isinstance(valor, float) and valor.is_integer()):
        return f"{int(valor):,}"
    if isinstance(valor, float):
        return f"{valor:,.2f}"
    return str(valor)

def renderizar_resultado(item, idx):
    """Mostra o resultado de uma pergunta (métrica, abas de gráfico/tabela/SQL)."""
    if item.get("bloqueio"):
        st.warning(f"🛡️ Consulta bloqueada pelo guardrail: {item['bloqueio']}")
        with st.expander("Ver detalhes técnicos"):
            st.code(item["sql"], language="sql")
        return
    if item.get("erro_llm"):
        st.error("Não foi possível gerar a consulta. Verifique sua cota no OpenRouter e tente novamente.")
        with st.expander("Ver detalhes técnicos"):
            st.write(item["info"])
        return
    if item.get("erro_sql"):
        st.error("Não consegui montar essa consulta. Tente reformular a pergunta.")
        with st.expander("Ver detalhes técnicos"):
            st.code(item["sql"], language="sql")
            st.write(item["erro_sql"])
        return

    df = item["df"]
    if df is None or df.empty:
        st.warning("A consulta foi executada com sucesso, mas não retornou nenhum dado.")
        with st.expander("Ver Query SQL Gerada"):
            st.code(item["sql"], language="sql")
        return

    st.caption(item["info"])

    # Resultado único (1 linha x 1 coluna) vira cartão de métrica
    if df.shape == (1, 1):
        st.metric(df.columns[0], formatar_metrica(df.iloc[0, 0]))
        with st.expander("Ver Query SQL Gerada"):
            st.code(item["sql"], language="sql")
        return

    aba_grafico, aba_tabela, aba_sql = st.tabs(["📊 Gráfico", "📋 Tabela", "🧾 SQL"])

    with aba_grafico:
        metricas = colunas_numericas(df)
        if len(df.columns) >= 2 and metricas and eixo_x_valido(df):
            y = metricas[0]
            if len(metricas) > 1:
                y = st.selectbox("Métrica do gráfico", metricas, key=f"metrica_{idx}")
            st.plotly_chart(desenhar_grafico(df, y), use_container_width=True, key=f"grafico_{idx}")
            if len(df) > MAX_BARRAS and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                st.caption(f"Exibindo as primeiras {MAX_BARRAS} linhas no gráfico.")
        else:
            st.info("Este resultado não tem um formato adequado para gráfico. Veja a aba Tabela.")

    with aba_tabela:
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config=configurar_colunas(df),
        )
        st.download_button(
            "⬇️ Baixar CSV", df.to_csv(index=False).encode("utf-8-sig"),
            file_name="resultado.csv", mime="text/csv", key=f"csv_{idx}",
        )

    with aba_sql:
        st.code(item["sql"], language="sql")

# --- PIPELINE DE UMA PERGUNTA ---
def processar_pergunta(pergunta, esquema):
    """Gera o SQL, valida (guardrail), executa e devolve um dicionário com tudo que a interface precisa."""
    item = {"pergunta": pergunta, "sql": None, "info": "", "df": None}

    with st.status("Analisando sua pergunta...", expanded=True) as status:
        st.write("🧠 Gerando consulta SQL...")
        sql, info, do_cache = gerar_sql(pergunta, esquema)
        item["info"] = info

        if not sql:
            item["erro_llm"] = True
            status.update(label="Falha ao gerar a consulta", state="error", expanded=False)
            return item

        if do_cache:
            st.toast("Resposta do cache ⚡")

        st.write("🛡️ Validando segurança da consulta...")
        ok, sql_limpo, motivo = validar_sql(sql)
        item["sql"] = sql_limpo
        if not ok:
            item["bloqueio"] = motivo
            status.update(label="Consulta bloqueada", state="error", expanded=False)
            return item

        st.write("⚙️ Executando no banco...")
        df, erro = executar_sql(sql_limpo)
        if erro:
            item["erro_sql"] = erro
            status.update(label="Erro ao executar a consulta", state="error", expanded=False)
            return item

        item["df"] = df
        # Só entra no cache o que passou no guardrail e executou sem erro
        if not do_cache:
            cache = carregar_cache()
            cache[pergunta.strip().lower()] = sql_limpo
            salvar_cache(cache)

        status.update(label="Análise concluída", state="complete", expanded=False)
    return item

# --- INTERFACE DE USUÁRIO (STREAMLIT) ---
st.set_page_config(page_title="CineData Analytics", page_icon="🎬", layout="wide")

if "historico" not in st.session_state:
    st.session_state.historico = []
if "pendente" not in st.session_state:
    st.session_state.pendente = None

esquema = obter_esquema()

# Barra lateral: exemplos clicáveis por categoria
with st.sidebar:
    st.header("💡 Perguntas de exemplo")
    st.caption("Clique para perguntar. Respostas já feitas saem do cache e não gastam cota da API.")
    for categoria, perguntas in EXEMPLOS.items():
        with st.expander(categoria):
            for i, texto in enumerate(perguntas):
                if st.button(texto, key=f"ex_{categoria}_{i}", use_container_width=True):
                    st.session_state.pendente = texto
    st.divider()
    st.caption(f"⚡ {len(carregar_cache())} consultas em cache")
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.historico = []
        st.rerun()

st.title("🎬 CineData Analytics - Assistente de Dados")
st.markdown("Faça perguntas em linguagem natural sobre nosso catálogo de filmes. O agente transformará sua pergunta em SQL e executará a consulta na nossa camada Gold.")

if esquema.startswith("Erro"):
    st.error(f"Não foi possível abrir o banco `{DB_FILE}`. Coloque o arquivo na mesma pasta do app.")
    with st.expander("Detalhes técnicos"):
        st.write(esquema)
    st.stop()

# Histórico da conversa
for idx, item in enumerate(st.session_state.historico):
    with st.chat_message("user"):
        st.write(item["pergunta"])
    with st.chat_message("assistant"):
        renderizar_resultado(item, idx)

# Nova pergunta (campo de chat ou clique na barra lateral)
nova = st.chat_input("Qual a sua pergunta? Ex: Quais os top 10 filmes com maior receita?")
if st.session_state.pendente:
    nova = st.session_state.pendente
    st.session_state.pendente = None

if nova:
    with st.chat_message("user"):
        st.write(nova)
    with st.chat_message("assistant"):
        resultado = processar_pergunta(nova, esquema)
        idx = len(st.session_state.historico)
        st.session_state.historico.append(resultado)
        renderizar_resultado(resultado, idx)
