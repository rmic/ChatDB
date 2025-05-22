import chainlit as cl
from chainlit import run_sync
from langchain.tools import BaseTool


class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name:str = "human"
    description:str = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )
    user_profile:str = ""
    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        if res:
            return res["content"]
        else:
            return "Human did not respond"
