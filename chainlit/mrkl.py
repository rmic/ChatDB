# from chainlit.prompt import Prompt, PromptMessage
import asyncio

from chainlit import Message, run_sync
from chainlit.input_widget import TextInput, Switch
from langchain import OpenAI, LLMMathChain, SerpAPIWrapper, PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType, OpenAIFunctionsAgent, load_tools
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
import os
import chainlit as cl
from langchain.graphs import Neo4jGraph
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.tools import PythonREPLTool, HumanInputRun, BaseTool
from pprint import pformat, pprint

os.environ["OPENAI_API_KEY"] = "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"


graph = Neo4jGraph(
    url="neo4j+s://26aef8a7.databases.neo4j.io",
    username="neo4j",
    password="rC7s6H6iL3PbQF7lXx6NyDxF3rB3sXBfyj7QSlLGE_s",
    database="neo4j"
)

class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["content"]

user_prompt_template = ChatPromptTemplate.from_template(
       """You are a BI assistant that answers a users' question about his data. When addressing the user, please make sure to use a language that corresponds the most to this user profile: {user_profile}
       Question: {question}"""
    )

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database. 
Schema:
{schema}
Instructions:
Execute one or more queries to answer the user's question
The following node types are present in the database:
Person: represent people aka. patients
ConditionOccurrence: represent condition or diseases diagnosed 
DrugOccurrence: Represent a drug or medication being prescribed to a patient
ProcedureOccurrence: represent a medical procedure carried out on a patient
Other node types and relationships exist in the database, please only generate cypher code that make use of existing node types and relationships
Note: Do make use of the existing relationships, especially between person and ConditionOccurrence, DrugOccurrence and ProcedureOccurrence, everytime you can, before attempting to build any external computation.
Start by analyzing the graph schema, then build your cypher query around it.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)


@cl.on_settings_update
async def settings_updated(settings):
    cl.user_session.set("settings", settings)
    cl.user_session.set("user_profile", settings["user_profile"])
    await Message(
        content=f"T'as changé tes settings, voilà les nouveaux : \n{pformat(settings['user_profile'], indent=4)}",
    ).send()


@cl.action_callback("le bouton")
async def le_bouton(action):
    await cl.Message(content=f"J'avais dit PAS TOUCHE B**** !!!").send()
    await cl.Message(content=f"T'as gagné, j'enlève le bouton maintenant !").send()
    await action.remove()


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="user_profile",
                label="User profile",
                initial="Talk to me like I'm 5 years old. I'm a big boy now."),
            Switch(id="human_input_allowed", label="Je me débrouille tout seul", initial=True),

        ]).send()
    cl.user_session.set('settings', settings)
    cl.user_session.set('user_profile', settings["user_profile"])

    actions = [
        cl.Action(name="le bouton", value="J'avais dit de ne pas toucher !!", description="DO NOT TOUCH me!")
    ]

    await cl.Message(content="Voici un bouton:", actions=actions).send()
    llm1 = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    # coai = ChatOpenAI(model_name="gpt-4", temperature=0)
    cypher_tool = GraphCypherQAChain.from_llm(
        llm1, graph=graph, verbose=True, prompt=CYPHER_GENERATION_PROMPT
    )

    inputs = {"profile": settings["user_profile"]}
    template = CYPHER_GENERATION_TEMPLATE

    tools = load_tools(["human"])

    def prompt_human(text):
        loop = asyncio.get_event_loop()
        coroutine = cl.Message(content=text).send()
        loop.run_until_complete(coroutine)
    def sync_human_input():
        loop = asyncio.get_event_loop()
        coroutine = get_human_input()
        result = loop.run_until_complete(coroutine)
        return result
    async def get_human_input():
        res = await cl.AskUserMessage(content="Please provide your answer ", timeout=30).send()
        if res:
            await cl.Message(
                content=f"Thank you, I'm going on with my thinking",
            ).send()
            return res['content']
        return None

    human = HumanInputRun(input_func=sync_human_input, prompt_func=prompt_human)
    tools = [
        Tool(
            name="Ask for clarification",
            func=HumanInputChainlit()._run,
            description="As a last resort, when you need to ask the human for clarifications"
        ),
        Tool(
            name="Intermediate Answer",
            func=cypher_tool.run,
            description="""
                Utilize this tool to search within the medical database, 
                specifically designed to find information about patients, their diseases and conditions, the procedures they have undergone and the medications they have been given.
                This specialized tool offers streamlined search capabilities
                to help you find the information you need with ease.
                """,
        )
    ]


    agent = initialize_agent(
        tools, llm1,  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, prompt=template, verbose=True,  memory=memory
    )

    from langchain.prompts import MessagesPlaceholder

    MEMORY_KEY = "chat_history"
    system_message = " "
    # prompt = OpenAIFunctionsAgent.create_prompt(
    #    extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
    # )
    # agent = OpenAIFunctionsAgent(llm=llm1, tools=tools, prompt=prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message):

    agent = cl.user_session.get("agent")  # type: AgentExecutor
    user_profile = cl.user_session.get("settings")["user_profile"]
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    prompt = user_prompt_template.format_messages(question=message, user_profile=user_profile)
    await cl.make_async(agent.run)(prompt, callbacks=[cb])
