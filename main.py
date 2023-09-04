import streamlit as st

st.title('🎈💬 ChatDB')

st.write('Hello world!')

user_input = st.text_input("Que voulez-vous savoir ?")
with st.sidebar:
    st.title('🤗💬 ChatDB')
    if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
        st.success('HuggingFace Login credentials already provided!', icon='✅')
        hf_email = st.secrets['EMAIL']
        hf_pass = st.secrets['PASS']
    else:
        hf_email = st.text_input('Enter E-mail:', type='password')
        hf_pass = st.text_input('Enter password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')
    #local_path = '/Users/rm/Library/Application Support/nomic.ai/GPT4All/ggml-gpt4all-l13b-snoozy.bin'
    #llm = GPT4All(model=local_path, verbose=True)

    #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


    #llm = OpenAI(temperature=0, verbose=True, openai_api_key=openai_api_key)
    #chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True, return_direct=True)

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
    #result = chain(user_input)
    #pprint(result)
    #st.markdown(result['result'])

