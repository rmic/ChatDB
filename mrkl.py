
from chainlit.input_widget import TextInput
from langchain import PromptTemplate, OpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType

from langchain.chat_models import ChatOpenAI
import os
import chainlit as cl
from langchain.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from pprint import pprint
from chatbot.human_input import HumanInputChainlit

from chatbot.memory import MyMemory
from chatbot.neo4j_tool import RBACGraphCypherQAChain
import yaml

os.environ["OPENAI_API_KEY"] = "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"

graph = Neo4jGraph(
    url="neo4j+s://26aef8a7.databases.neo4j.io",
    username="neo4j",
    password="rC7s6H6iL3PbQF7lXx6NyDxF3rB3sXBfyj7QSlLGE_s",
    database="neo4j"
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

# oai = le modèle qui va générer le code cypher pour la db neo4j
oai = OpenAI(model_name="gpt-4", temperature=0)
template = CYPHER_GENERATION_TEMPLATE

# cypher tool = l'outil qui utilise le modèle (et qui va check les permissions)
cypher_tool = RBACGraphCypherQAChain.from_llm(
        oai, graph=graph, verbose=True, prompt=CYPHER_GENERATION_PROMPT
)


@cl.on_settings_update
async def settings_updated(settings):
    """
    Executed when chat settings are updated, using the button next to the input box
    :param settings: the new settings
    """
    original_token = cl.user_session.get("settings")["user_token"]
    cl.user_session.set("settings", settings)

    cl.user_session.set("user_infos","")


    if settings["user_token"] != original_token:
        with open('users.yaml', 'r') as f:
            users = yaml.safe_load(f)

        for name, user in users.items():
            if user["token"].strip() == settings["user_token"].strip():
                pprint(user)
                cl.user_session.set("username", name)
                cl.user_session.set("user_roles", user["roles"] )
                cypher_tool.user_roles = user["roles"]
                with open('permissions.yaml', 'r') as f:
                    permissions = yaml.safe_load(f)

                denied = []
                allowed = []
                if permissions:
                    for role in user["roles"]:
                        if role in permissions.keys():
                            if "allow" in permissions[role]:
                                allowed.extend(permissions[role]["allow"])
                            if "deny" in permissions[role]:
                                denied.extend(permissions[role]["deny"])

                    cypher_tool.user_roles = user["roles"]
                    cypher_tool.user_allowed = list(set(allowed))
                    cypher_tool.user_denied = list(set(denied))

                hit = cl.user_session.get("human_input_tool")
                hit.user_profile = settings["user_profile"]
                cl.user_session.set("human_input_tool", hit)
                cl.user_session.set("user_profile", user["default_profile"])
                await cl.Message(content=f"Hello {name} !").send()
                return

        await cl.Message(content=f"Meeh, looks like I did not find your token in our users file").send()
    else:
        cl.user_session.set("user_profile", settings["user_profile"])



user_prompt_template = ChatPromptTemplate.from_messages([

    SystemMessage(
        content=(
                "You are a BI assistant that help to find answers to a users' questions about his data. When addressing the user, please make sure to use a language that corresponds the most to this user profile"
        )
    ),
    HumanMessagePromptTemplate.from_template("{question}"),
])



@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="user_token",
                label="Secret Token",
                initial=""),
            TextInput(
                id="user_profile",
                label="User profile",
                initial=""),
        ]).send()
    cl.user_session.set('settings', settings)
    cl.user_session.set('human_input_tool', HumanInputChainlit())
    cl.user_session.set('user_profile', settings["user_profile"])
    cl.user_session.set('user_infos', "")

    # llm1 = le modèle qui va servir pour l'agent conversationnel
    llm1 = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)

    # Les outils à disposition de l'agent pour répondre aux questions
    tools = [
        Tool(
            name="Ask for clarification",
            func=cl.user_session.get("human_input_tool")._run,
            description="Utilize this tool when you need to ask the human for clarifications"
        ),

        Tool(
             name="Intermediate Answer",
             func=cypher_tool.run,
             description="""
                 Utilize this tool to search within the medical database,
                 specifically designed to find information about patients, their diseases and conditions, the procedures they have undergone and the medications they have been given.
                 This specialized tool offers streamlined search capabilities
                 to help you find the information you need with ease.
                 """
        )
    ]

    # L'historique du chat
    memory = MyMemory(
        memory_key="chat_history", return_messages=True, )


    # Le fameux agent conversationnel (on peut changer le type d'agent, ça change un peu la façon dont il répond).
    agent = initialize_agent(
        tools, llm1,  agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        prompt=user_prompt_template, streaming=True,
        verbose=True,  memory=memory
    )

    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message):
    if cl.user_session.get("user_roles") is not None:
        agent = cl.user_session.get("agent")  # type: AgentExecutor
        user_profile = cl.user_session.get("settings")["user_profile"]

        # Idéalement il faudrait juste aller injecter le user profile dans le system message
        # mais c'était du chipotage donc je le redéfinis complètement ici. C'est pas propre.
        user_prompt_template = ChatPromptTemplate.from_messages([

            SystemMessage(
                content=(
                        "You are a BI assistant that help to find answers to a users' questions about his data. When addressing the user, please make sure to use a language that corresponds the most to this user profile: " + user_profile
                )
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])

        prompt = user_prompt_template.format_messages(question=message, user_profile=user_profile)
        response = await cl.make_async(agent.run)(prompt, callbacks=[cl.LangchainCallbackHandler()])
        await cl.Message(content=response).send()
    else:
        await cl.Message(content="Please provide a valid token in to start (hint: go to settings, then fill in the 'secret token' field").send()

