import chainlit as cl
from chainlit import run_sync
from langchain.tools import BaseTool


class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )
    user_profile = ""
    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        agent = cl.user_session.get("agent")
        prompt = f"""The system needs to ask the user for more information. 
                     Can you format the following query according to the user's profile and preference ? 
                     Note: Stick as best as you can to the meaning of the original query, 
                     do not add any other information, explanation or apologies, 
                     but make sure it is tailored to the user profile and your previous interactions. 
                     Do NOT use another tool to perform this operation. 
                     User profile:{self.user_profile} 
                     Query:{query}
                """
        user_q = agent.run(prompt)
        res = run_sync(cl.AskUserMessage(content=user_q).send())
        if res:
            return res["content"]
        else:
            return "Human did not respond"
