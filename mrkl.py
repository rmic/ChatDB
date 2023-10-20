from chainlit.input_widget import TextInput, Select, Switch
from langchain import PromptTemplate, OpenAI, ConversationChain
from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType, ConversationalAgent, ZeroShotAgent
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI
import os
import chainlit as cl
from langchain.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, OutputParserException
from pprint import pprint
from chatbot.human_input import HumanInputChainlit
from chatbot.prompts import create_prompt
from chatbot.memory import MyMemory, ExtendedConversationEntityMemory
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
oai = ChatOpenAI(model_name="gpt-4", temperature=0)
template = CYPHER_GENERATION_TEMPLATE

# cypher tool = l'outil qui utilise le modèle (et qui va check les permissions)
cypher_tool = RBACGraphCypherQAChain.from_llm(
    oai, graph=graph, verbose=True, prompt=CYPHER_GENERATION_PROMPT
)


@cl.action_callback("Ask !")
async def on_question(action):
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

    prompt = user_prompt_template.format_messages(question=action.value, user_profile=user_profile)
    response = await cl.make_async(agent.run)(prompt, callbacks=[cl.LangchainCallbackHandler()])
    await cl.Message(content=response).send()


async def suggest_initial_questions(user_profile):
    agent = cl.user_session.get("agent")


    response = await cl.make_async(agent.run)(
        f"given what you know about the database, and the following user profile : \n{user_profile}\n\nPlease provide 3 proposals for analytical questions about the data that this type of user might potentially be interested in. Write each question on one single line, do not add anything to the text, it should only contain your questions.")

    for line in response.split('\n'):
        actions = [
            cl.Action(name="Ask !", value=line, description="Ask!")
        ]
        await cl.Message(content=line, actions=actions).send()
@cl.on_settings_update
async def settings_updated(settings):
    """
    Executed when chat settings are updated, using the button next to the input box
    :param settings: the new settings
    """
    #original_token = cl.user_session.get("settings")["user_token"]
    cl.user_session.set("settings", settings)

    cl.user_session.set("user_infos", "")

    role = settings["role"] if "role" in settings.keys() else None
    previous_role = cl.user_session.get('role')
    if role is not None and previous_role != role:
        cl.user_session.set('role', role)

        with open('permissions.yaml', 'r') as f:
            permissions = yaml.safe_load(f)

            denied = []
            allowed = []
            if permissions:
                if role in permissions.keys():
                    if "allow" in permissions[role]:
                        allowed.extend(permissions[role]["allow"])
                    if "deny" in permissions[role]:
                        denied.extend(permissions[role]["deny"])

                cypher_tool.user_roles = [role]
                cypher_tool.user_allowed = list(set(allowed))
                cypher_tool.user_denied = list(set(denied))

        if settings["user_profile"]:
            cl.user_session.set('user_profile', settings['user_profile'])
        else:
            roles = load_roles()
            user_profile = roles.get('roles').get(role)
            cl.user_session.set('user_profile', user_profile)
            settings["user_profile"] = user_profile
            cl.user_session.set('settings', settings)
            await cl.Message(
                content=f'Your new user profile is defined with the default setting as : {user_profile}').send()
            if suggestions_are_enabled():
                await suggest_initial_questions(user_profile)



    # if settings["user_token"]:
    #     with open('users.yaml', 'r') as f:
    #         users = yaml.safe_load(f)
    #
    #     for name, user in users.items():
    #         if user["token"].strip() == settings["user_token"].strip():
    #             pprint(user)
    #             cl.user_session.set("username", name)
    #             cl.user_session.set("user_roles", user["roles"])
    #             cypher_tool.user_roles = user["roles"]
    #             with open('permissions.yaml', 'r') as f:
    #                 permissions = yaml.safe_load(f)
    #
    #             denied = []
    #             allowed = []
    #             if permissions:
    #                 for role in user["roles"]:
    #                     if role in permissions.keys():
    #                         if "allow" in permissions[role]:
    #                             allowed.extend(permissions[role]["allow"])
    #                         if "deny" in permissions[role]:
    #                             denied.extend(permissions[role]["deny"])
    #
    #                 cypher_tool.user_roles = user["roles"]
    #                 cypher_tool.user_allowed = list(set(allowed))
    #                 cypher_tool.user_denied = list(set(denied))
    #
    #             hit = cl.user_session.get("human_input_tool")
    #             hit.user_profile = settings["user_profile"]
    #
    #             cl.user_session.set("human_input_tool", hit)
    #             cl.user_session.set("user_profile", user["default_profile"])
    #             cl.user_session.set("settings")["user_profile"] = user["default_profile"]
    #             await cl.Message(content=f"Hello {name} !").send()
    #             return
    #
    #     await cl.Message(content=f"Meeh, looks like I did not find your token in our users file").send()
    # else:
    #     cl.user_session.set("user_profile", settings["user_profile"])




def load_roles():
    with open('roles.yaml', 'r') as f:
        roles = yaml.safe_load(f)

    return roles
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
memory = ExtendedConversationEntityMemory(llm=gpt4, return_messages=True, extra_variables=["entities", "user_profile", "agent_scratchpad"])
tool_names = []
@cl.on_chat_start
async def start():
    roles= load_roles()
    role_names = list(roles['roles'].keys())

    settings = await cl.ChatSettings(
        [
            Select(
                id="role",
                label="User role",
                values=role_names,
            ),
           # TextInput(
           #     id="user_name",
           #     label="Your name",
           #     initial=""),
            TextInput(
                id="user_profile",
                label="User profile",
                placeholder="This will be replaced with your default profile",
            ),
            Switch(id="generate_suggestions", label="Generate suggestions", initial=False)
        ]).send()
    cl.user_session.set('settings', settings)
    cl.user_session.set('human_input_tool', HumanInputChainlit())
    cl.user_session.set('user_profile', settings["user_profile"])
    cl.user_session.set('user_infos', "")

    # gpt4 = le modèle qui va servir pour l'agent conversationnel


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
        ),
    ]



    # Le fameux agent conversationnel (on peut changer le type d'agent, ça change un peu la façon dont il répond).


    llm_chain = ConversationChain(memory=memory, prompt=create_prompt(tools), llm=gpt4)
    tool_names = [tool.name for tool in tools]

    #conv_agent = ConversationalAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent2 = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
         agent=agent2,
         tools=tools,
         streaming=True,
         verbose=True,
    )

    #agent_executor = initialize_agent(
    #    tools, gpt4, agent=AgentType.C,
    #    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    #    streaming=True,
    #    verbose=True,
    #    memory=memory
    #)

    cl.user_session.set("agent", agent_executor)

def suggestions_are_enabled():
    return cl.user_session.get('settings').get('generate_suggestions')


@cl.on_message
async def main(message):
    agent_executor = cl.user_session.get("agent")  # type: AgentExecutor
    user_profile = cl.user_session.get("settings")["user_profile"]

    try:
        response = await cl.make_async(agent_executor.run)({"input":message, "user_profile":user_profile},callbacks=[cl.LangchainCallbackHandler()])
    except OutputParserException as e:
        if "Final Answer:" in str(e):
            position = str(e).find("Final Answer:")
            response = str(e)[position+16:]
        else:
            response = str(e)

    await cl.Message(content=response, disable_human_feedback=True).send()

    if suggestions_are_enabled():
        question="based on the previous questions, please generate a few additional questions related to the same topic if relevant. If you feel like this is not the time to do so, just output a nice message to tell the user you are there to help."
        response = await cl.make_async(agent_executor.run)({"input": question, "user_profile": user_profile},
                                                           callbacks=[cl.LangchainCallbackHandler()])

        await cl.Message(content=response).send()