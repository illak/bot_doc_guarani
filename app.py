import streamlit as st
#from openai import OpenAI

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

import re
from bs4 import BeautifulSoup

import os
import yaml

#config = yaml.safe_load(open("config.yml"))

#os.environ["OPENROUTER_API_KEY"] = config["OPENROUTER_API_KEY"]
#os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]
#os.environ["LANGCHAIN_TRACING_V2"] = str(config["LANGCHAIN_TRACING_V2"]).lower()
#os.environ["LANGCHAIN_ENDPOINT"] = config["LANGCHAIN_ENDPOINT"]
#os.environ["LANGCHAIN_PROJECT"] = config["LANGCHAIN_PROJECT"]
#os.environ["LANGCHAIN_HUB_API_KEY"] = config["LANGCHAIN_API_KEY"]
#os.environ["LANGCHAIN_HUB_API_URL"] = config["LANGCHAIN_HUB_API_URL"]
#os.environ["SEARCHAPI_API_KEY"] = config["SEARCHAPI_API_KEY"]
#os.environ["GOOGLE_GEMINI_API_KEY"] = config["GOOGLE_GEMINI_API_KEY"]
#os.environ["GOOGLE_API_KEY"] = config["GOOGLE_GEMINI_API_KEY"]

LOGO_GUARANI = "imgs/logo.png"
LOGO_LINKEDIN = "imgs/logo_linkedin.png"
LOGO_GITHUB = "imgs/logo_github.png"
LOGO_X = ""

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_documents():

    url= "https://documentacion.siu.edu.ar/wiki/SIU-Guarani/Version3.21.0/documentacion_de_las_operaciones"
    url2 = "https://documentacion.siu.edu.ar/wiki/SIU-Guarani/Version3.21.0/principales_circuitos_funcionales"
    url3 = "https://documentacion.siu.edu.ar/wiki/SIU-Guarani/Version3.21.0"

    loader = RecursiveUrlLoader(
        url, 
        max_depth=4,
        #extractor=bs4_extractor
        )

    docs = loader.load()

    return docs


def transform_documents(docs):

    md = MarkdownifyTransformer()
    converted_docs = md.transform_documents(docs)

    return converted_docs


def generate_chunks(docs):

    headers_to_split_on = [
        ("#", "Header 1"),
        #("##", "Header 2"),
        #("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        #return_each_line=True
        )
    
    chunks = []

    for doc in docs:
        chunks = chunks + markdown_splitter.split_text(doc.page_content)

    return chunks


def get_retriever(chunks, gemini_api_key):

    #llm = ChatGoogleGenerativeAI(model="gemini-pro")
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                              google_api_key=gemini_api_key)
    # 3. Crear embeddings
    #embeddings = HuggingFaceEmbeddings()
    # 4. Almacenar embeddings en FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Guardamos BBDD vectorial
    # Guardar el índice en disco
    index_name = "guarani-index"
    vector_store.save_local("content/", index_name=index_name)

    retriever = vector_store.as_retriever()

    return retriever


def load_retriever(gemini_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                              google_api_key=gemini_api_key)

    # Cargamos el índice
    index_name = "guarani-index"
    loaded_vector_store = FAISS.load_local("content/", embeddings, index_name=index_name, allow_dangerous_deserialization=True)

    retriever = loaded_vector_store.as_retriever()

    return retriever


def get_prompt():
    prompt_template = """
Utiliza el siguiente contexto extraído de documentación online como tu base de conocimiento:
\n
=======================
{context}
=======================
\n
A partir del contexto responda la pregunta del usuario de manera detallada y amable.
Responda con enlaces y urls a la documentación siempre que sea posible.
Si no encuentra la respuesta en el contexto responde lo siguiente:
"No encontré información la respecto en mi base de conocimiento".
Siempre agregar al final de la respuesta lo siguiente:

"**💡Recordá que podés encontrar más información en la [wiki de Guaraní](https://documentacion.siu.edu.ar/wiki/SIU-Guarani/version3.22.0) 
y en el [foro de la comunidad](https://foro.comunidad.siu.edu.ar/c/siu-guarani/95).**"

Pregunta: {question}

Respuesta:\n

"""
    prompt = PromptTemplate.from_template(prompt_template)

    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_qa_chain(retriever, llm, prompt):
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


def init_context(gemini_api_key):
    try:
        print("Obteniendo datos guardados...\n")
        retriever = load_retriever(gemini_api_key)
    except:
        print("Falló carga de datos guardados en disco...\n")
        print("Realizando carga inicial de documentos (esto puede demorar)...\n")
        docs = load_documents()
        transformed_documents = transform_documents(docs)
        chunks = generate_chunks(transformed_documents)
        retriever = get_retriever(chunks, gemini_api_key)

    return retriever
        

with st.sidebar:
    st.markdown("Para poder usar el chatbot deberá obtener una clave API de Google Gemini y agregarla debajo:")
    gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
    "[Obtener clave API para Gemini](https://ai.google.dev/)"

    st.divider()
    st.markdown("Desarrollado por: *Lic. Illak Zapata*")

    columns = st.sidebar.columns(6)

    with columns[1]:
        st.write("""<div style="width:100%;text-align:center;"><a href="https://www.linkedin.com/in/illakzapata/" style="float:center"><img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="22px"></img></a></div>""", unsafe_allow_html=True)

    with columns[2]:
        st.write("""<div style="width:100%;text-align:center;"><a href="https://github.com/illak" style="float:center"><img src="https://img.icons8.com/material-outlined/48/000000/github.png" width="22px"></img></a></div>""", unsafe_allow_html=True)

    with columns[3]:
        st.write("""<div style="width:100%;text-align:center;"><a href="https://illak-blog.netlify.app/" style="float:center"><img src="https://www.freeiconspng.com/uploads/website-icon-11.png" width="22px"></img></a></div>""", unsafe_allow_html=True)


st.title("Chatbot SIU-Guaraní - Gestión 🤖")
st.subheader("Documentación de las operaciones ([versión 3.21.0](https://documentacion.siu.edu.ar/wiki/SIU-Guarani/Version3.21.0/documentacion_de_las_operaciones)).")
with st.expander("Más información 🔽"):
    st.markdown("Bienvenido/a al chatbot de consultas sobre operaciones del módulo de Gestión\n\
del sistema Guaraní. Se recomienda hacer preguntas detalladas y completas\n\
para obtener una mayor eficacia en las respuestas.")
    st.markdown("""
    Si el BOT no encuentra resultados, pruebe con distintas versiones de una misma pregunta, por ejemplo:
    - *¿Cómo puedo cambiar a un alumno de comisión?*
    - *¿Cómo puedo mover a un alumno de una comisión a otra?*
                
⚠️**Recordá que esta herramienta es experimental y puede devolver información que es poco precisa o incorrecta, por lo tanto verificá siempre
con la documentación oficial y no te quedes con la última palabra del asistente.**
    """)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! Estoy aquí para asistirte. Realizá tu pregunta."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Su consulta"):
    if not gemini_api_key:
        st.info("Por favor agregue su clave API de Google Gemini.")
        st.stop()

    retriever = init_context(gemini_api_key)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=gemini_api_key)
    
    prompt_guarani = get_prompt()
    
    qa_chain = get_qa_chain(retriever, llm, prompt_guarani)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    #response = qa_chain.invoke(prompt)

    #msg = response
    with st.spinner("Estoy buscando información en mi base de conocimiento..."):
        response = st.write_stream(qa_chain.stream(prompt))
        
        st.session_state.messages.append({"role": "assistant", "content": response})