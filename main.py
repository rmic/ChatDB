from langchain import OpenAI, SQLDatabase, SQLDatabaseChain, PromptTemplate, LLMChain, LlamaCpp, VectorDBQA
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st
import psycopg2
import openai
from pprint import pprint

from langchain.agents import initialize_agent, AgentType
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import FAISS
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#
# Load indexes with relevant data
# i.e (medications, diseases, etc.)
loader = WebBaseLoader(["http://ohdsi.github.io/CommonDataModel/cdm54.html"])
pages = loader.load()

text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)
# then FAISS
openai_api_key = "set-api-key-here"
#embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#docs_db = FAISS.from_documents(docs, embeddings)


db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://user:password@hostname:5432/db_name",
)


template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager

st.title("💬ChatDB")
user_input = st.text_input("Que voulez-vous savoir ?")
if user_input:
    local_path = '/Users/rm/Library/Application Support/nomic.ai/GPT4All/ggml-gpt4all-l13b-snoozy.bin'
    llm = GPT4All(model=local_path, verbose=True)

    #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


    #llm = OpenAI(temperature=0, verbose=True, openai_api_key=openai_api_key)
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True, return_direct=True)

    #search = DuckDuckGoSearchRun()
    #tools = [
    #    Tool(
    #        name="Intermediate Answer",
    #        func=search.run,
    #        description="useful for when you need to ask with search"
    #    )
    #]
    #chain = LLMChain(llm=llm, prompt=prompt)

    #relevant_docs = docs_db.similarity_search(user_input)
    #qa = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docs_db)

    #self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
    #result = self_ask_with_search.run(user_input)
    #result = qa({'query':user_input})
    result = chain(user_input)
    pprint(result)
    st.markdown(result['result'])

