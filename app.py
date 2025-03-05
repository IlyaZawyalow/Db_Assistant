import os
import chainlit as cl
from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableConfig, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from datetime import datetime
import aioodbc

API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_USER = os.environ.get("DB_USER")

PDF_FILE_PATH = "./postgres-A4.pdf"

# Инициализация Chroma
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
persist_directory = "./chroma_db"



def load_or_process_pdf():
    if not os.path.exists(PDF_FILE_PATH):
        raise FileNotFoundError(f"PDF file not found at {PDF_FILE_PATH}")
    
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        ).as_retriever()
    
    loader = PyPDFLoader(PDF_FILE_PATH)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(pages)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore.as_retriever()

store = {}
llm = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0,
    mistral_api_key=API_KEY,
    mistral_api_base=BASE_URL
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert assistant for database-related QA. Your role is to help users answer questions about databases and generate SQL queries "
            "based on the provided database structure. Use the provided context to deliver clear and concise answers. "
            "If the context does not provide enough information, simply say 'I don't know'. "
            "Keep your response as short and direct as possible. "
            "Please communicate with me in Russian. "
            "Context: {context}\n"
        )
    ),
    MessagesPlaceholder("history"),
    ("human", "Question: {question}")
])

trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=2000,
    start_on="human",
    end_on="human",
    include_system=True,
    allow_partial=False
)

def get_history_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

format_docs_runnable = RunnableLambda(
    lambda docs: "\n\n".join(d.page_content for d in docs)
).with_config(config=RunnableConfig(run_name="format documents"))

@cl.on_chat_start
async def init_chat():
    try:
        retriever = load_or_process_pdf()
        cl.user_session.set("retriever", retriever)
        await cl.Message(content="Контекст успешно загружен! Можете задавать вопросы.").send()
    except FileNotFoundError:
        await cl.Message(content=f"Ошибка: Файл {PDF_FILE_PATH} не найден!").send()
        raise
    except Exception as e:
        await cl.Message(content=f"Ошибка загрузки контекста: {str(e)}").send()
        raise

@cl.on_message
async def main(message: cl.Message):
    user_session_id = cl.user_session.get("id")
    retriever = cl.user_session.get("retriever")
    
    if not retriever:
        await cl.Message(content="Контекст не загружен!").send()
        return
    
    question = message.content

    # Исправленная цепочка
    chain = (
        RunnableParallel({
            "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs_runnable,
            "question": lambda x: x["question"],
            "history": lambda x: x.get("history", [])
        })
        | prompt
        | trimmer
        | llm
        | StrOutputParser()
    )

    final_chain = RunnableWithMessageHistory(
        chain,
        get_history_by_session_id,
        input_messages_key="question",
        history_messages_key="history"
    )
    
    msg = cl.Message(content="")
    async for chunk in final_chain.astream(
        {"question": question},
        config=RunnableConfig(configurable={"session_id": user_session_id})
    ):
        await msg.stream_token(chunk)
    await msg.send()

async def create_connection(database: str) -> aioodbc.Connection | None:
    try:
        conn = await aioodbc.connect(
            dsn=f"Driver={{PostgreSQL ODBC Driver(UNICODE)}};"
            f"Server={DB_HOST};"
            f"Port={DB_PORT};"
            f"Database={database};"
            f"Uid={DB_USER};"
            f"Pwd={DB_PASSWORD};"
        )
        return conn
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return None
@tool
async def get_databases() -> list[str]:
    """Получить список всех баз данных"""
    conn = None
    try:
        conn = await create_connection("postgres")
        if not conn:
            return []

        async with conn.cursor() as cursor:
            await cursor.execute("""
                SELECT datname 
                FROM pg_database 
                WHERE datistemplate = false 
                AND datname NOT IN ('postgres')
            """)
            return [row[0] for row in await cursor.fetchall()]
            
    except Exception as e:
        print("Ошибка при получении списка БД:", e)
        return []
    finally:
        if conn: await conn.close()

@tool
async def get_database_structure(db_name: str) -> dict:
    """Получить структуру базы данных"""
    structure = {"database": db_name, "tables": []}
    conn = None
    
    try:
        conn = await create_connection(db_name)
        if not conn:
            return structure

        async with conn.cursor() as cursor:
            # Получаем список таблиц
            await cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                AND table_type = 'BASE TABLE'
            """)
            tables = [row[0] for row in await cursor.fetchall()]

            for table in tables:
                table_info = {
                    "table_name": table,
                    "columns": [],
                    "primary_keys": [],
                    "foreign_keys": []
                }

                # Получаем информацию о столбцах
                await cursor.execute(f"""
                    SELECT 
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        c.column_default,
                        c.character_maximum_length,
                        c.numeric_precision,
                        c.numeric_scale
                    FROM information_schema.columns c
                    WHERE c.table_name = '{table}'
                    ORDER BY c.ordinal_position
                """)
                
                # Обрабатываем столбцы
                for col in await cursor.fetchall():
                    column = {
                        "name": col[0],
                        "type": col[1],
                        "nullable": col[2] == 'YES',
                        "default": col[3],
                        "max_length": col[4],
                        "numeric_precision": col[5],
                        "numeric_scale": col[6]
                    }
                    table_info["columns"].append(column)

                # Получаем первичные ключи
                await cursor.execute(f"""
                    SELECT kcu.column_name 
                    FROM information_schema.key_column_usage kcu
                    JOIN information_schema.table_constraints tc
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = '{table}'
                        AND tc.constraint_type = 'PRIMARY KEY'
                """)
                table_info["primary_keys"] = [row[0] for row in await cursor.fetchall()]

                # Получаем внешние ключи
                await cursor.execute(f"""
                    SELECT 
                        kcu.column_name,
                        ccu.table_name AS foreign_table,
                        ccu.column_name AS foreign_column
                    FROM information_schema.key_column_usage kcu
                    JOIN information_schema.constraint_column_usage ccu
                        ON ccu.constraint_name = kcu.constraint_name
                    WHERE kcu.table_name = '{table}'
                        AND kcu.constraint_name IN (
                            SELECT constraint_name 
                            FROM information_schema.table_constraints 
                            WHERE constraint_type = 'FOREIGN KEY'
                        )
                """)
                table_info["foreign_keys"] = [
                    {
                        "column": row[0],
                        "references": f"{row[1]}({row[2]})"
                    } for row in await cursor.fetchall()
                ]

                structure["tables"].append(table_info)

        return structure

    except Exception as e:
        print(f"Ошибка при получении структуры БД {db_name}: {e}")
        return {}
    finally:
        if conn: await conn.close()