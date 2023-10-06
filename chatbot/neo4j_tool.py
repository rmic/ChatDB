"""Question answering over a graph."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

INTERMEDIATE_STEPS_KEY = "intermediate_steps"
import yaml

def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text

def is_user_allowed(denied, allowed, nodes_and_edges):
    print(nodes_and_edges)
    print(denied)
    print(allowed)
    for item in nodes_and_edges.split("\n"):
        if (item in denied) and not (item in allowed):
            print(f"Data access denied because {item} is explicitly denied ({item in denied}) and not allowed ({item in allowed})")
            print(denied)
            print(allowed)
            return False
        elif (item in denied) and (item in allowed):
            print(f"{item} was denied but explicitely allowed elsewhere")
        elif (item in allowed):
            print(f"{item} was explicitely allowed")
        else:
            print(f"nothing was specified about {item}")

    return True

class RBACGraphCypherQAChain(Chain):
    """Chain for question-answering against a graph by generating Cypher statements."""

    graph: Neo4jGraph = Field(exclude=True)
    cypher_generation_chain: LLMChain
    qa_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 10
    """Number of results to return from the query"""
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the graph directly."""
    user_roles = []
    user_allowed = []
    user_denied = []

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    @property
    def _chain_type(self) -> str:
        return "graph_cypher_chain"

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
        cypher_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        **kwargs: Any,
    ) -> RBACGraphCypherQAChain:
        """Initialize from LLM."""

        if not cypher_llm and not llm:
            raise ValueError("Either `llm` or `cypher_llm` parameters must be provided")
        if not qa_llm and not llm:
            raise ValueError("Either `llm` or `qa_llm` parameters must be provided")
        if cypher_llm and qa_llm and llm:
            raise ValueError(
                "You can specify up to two of 'cypher_llm', 'qa_llm'"
                ", and 'llm', but not all three simultaneously."
            )

        qa_chain = LLMChain(llm=qa_llm or llm, prompt=qa_prompt)
        cypher_generation_chain = LLMChain(llm=cypher_llm or llm, prompt=cypher_prompt)

        return cls(
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]

        intermediate_steps: List = []

        generated_cypher = self.cypher_generation_chain.run(
            {"question": question, "schema": self.graph.get_schema}, callbacks=callbacks
        )

        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)
        nodes_and_edges = self.qa_chain.run(
            {"question": f'Output the list of node and edge types that are referenced in the cypher statement below. Please output a list of node types then edge types, each element on a separate line, no bullets. Do not output anything more than the node and edge types. The Cypher statement: {generated_cypher}', 'context': ""}
        )

        if not is_user_allowed(self.user_denied, self.user_allowed, nodes_and_edges):
            intermediate_steps.append({"check_users_permissions": "denied"})
            final_result = "Stop executing and explain to the user that he does not have access to the information requested due to insufficient users permissions."

        else:
            intermediate_steps.append({"check_users_permissions": "allowed"})
            _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                generated_cypher, color="green", end="\n", verbose=self.verbose
            )

            intermediate_steps.append({"query": generated_cypher})

            # Retrieve and limit the number of results
            try:
                context = self.graph.query(generated_cypher)[: self.top_k]
            except Exception as err:
                if "not valid" in str(err):
                    # The model probably made a mistake, let's ask him to fix that
                    print("Ugh, we made a mistake, we should fix that")
                    self.return_direct = True
                    context = "The generated cypher code was invalid. This is a known bug and we are very sorry. " \
                              "Our best code monkeys are on the case !"

            if self.return_direct:
                final_result = context
            else:
                _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
                _run_manager.on_text(
                    str(context), color="green", end="\n", verbose=self.verbose
                )

                intermediate_steps.append({"context": context})

                result = self.qa_chain(
                    {"question": question, "context": context},
                    callbacks=callbacks,
                )
                final_result = result[self.qa_chain.output_key]

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
