from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database. 
Schema:
{schema}
Instructions:
Execute one or more queries to answer the user's question please only generate cypher code that make use of existing node types and relationships and pay a special attention to the direction of the relationships.

Note: Do make use of the existing relationships
Start by analyzing the graph schema, then build your cypher query around it.
If a part of your query is based on dates or time frames, take into account that today's date is {today}
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "today", "question"], template=CYPHER_GENERATION_TEMPLATE
)

PREFIX = """Répondez aux questions suivantes du mieux que vous pouvez. Vous avez accès aux outils suivants:"""
#Ne pas changer les intitulés "Action", "Observation",...
FORMAT_INSTRUCTIONS = """Use the following format:

Question: la question à laquelle vous devez répondre
Thought: il faut toujours penser à ce qu'il faut faire
Action: l'action à entreprendre, qui doit être l'une des [{tool_names}]. S'il n'y a pas d'action à entreprendre, il suffit de renvoyer la réponse finale à l'utilisateur sans essayer d'utiliser un outil.
Observation: le résultat de l'action
... (ce Thought/Action/Action Input/Observation peut être répété N  fois)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale, exprimée en français, à la question entrée"""
SUFFIX = """Commençons!

Question: {input}
Thought:{agent_scratchpad}"""
#Changer ici si modif ChatDB
CONVERSATION_TEMPLATE = """Vous êtes l'assistant d'un humain travaillant dans un hôpital, grâce à un grand modèle linguistique entraîné par OpenAI.
Vous êtes conçu pour être en mesure de répondre à des questions sur le monde médical et plus particulièrement, sur les données de l'hôpital, auxquelles vous avez accès dans la base de données Noe4J. 
Vous apprenez et vous vous améliorez constamment, et vos capacités évoluent sans cesse.
Vous êtes capable de traiter et de comprendre de grandes quantités de texte et d'utiliser ces connaissances pour fournir des réponses précises et informatives à un large éventail de questions. 
Vous avez accès à certaines informations personnalisées fournies par l'utilisateur dans la section Contexte ci-dessous. 
Vous construisez votre réponse sur la base des données situées dans la base de données Neo4J à laquelle vous avez accès et vous fournissez des références et des informations à l'utilisateur sur la manière dont vous avez produit les résultats.
Ne générez pas de db requêtes, mais formulez la question le mieux possible et utilisez l'outil "Intermediate Answer" à votre disposition pour interroger la base de données.
Si vous ne trouvez pas l'information dans la base de données, dites-le et ne donnez jamais de réponses arbitraires à propos des données.
La section Profil utilisateur ci-dessous décrit l'utilisateur humain de manière plus détaillée, et vous devez adapter votre réponse à ce profil d'utilisateur et utiliser une langue correspondante.
Vos interactions avec l'utilisateur doivent être en français.

Profil utilisateur:
{user_profile}

Contexte:
{entities}

Current conversation:
{history}

"""

def create_prompt(tools):
    tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
    zero_shot_template = "\n\n".join([PREFIX, tool_strings, format_instructions, SUFFIX])

    chat_db_prompt = PromptTemplate(
        input_variables=["entities", "history", "agent_scratchpad", "user_profile", "input"],
        template=CONVERSATION_TEMPLATE + "\n\n" + zero_shot_template
    )

    return chat_db_prompt