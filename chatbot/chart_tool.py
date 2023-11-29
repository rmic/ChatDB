"""Question answering over a graph."""
from __future__ import annotations
import json
import logging
from urllib.parse import urlencode, quote_plus
import re
from typing import Any, Dict, List, Optional
import datetime
import chainlit as cl
import requests
from chainlit import run_sync
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from .prompts import CHART_GENERATION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

INTERMEDIATE_STEPS_KEY = "intermediate_steps"
import yaml


class ChartChain(Chain):
    """Chain for question-answering against a graph by generating Cypher statements."""


    chartjs_generation_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False


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
        return "chartjs_chain"

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        prompt: BasePromptTemplate = CHART_GENERATION_PROMPT,
        chart_llm: Optional[BaseLanguageModel] = None,
        **kwargs: Any,
    ) -> ChartChain:
        """Initialize from LLM."""
        chartjs_generation_chain = LLMChain(llm=chart_llm or llm, prompt=CHART_GENERATION_PROMPT)

        return cls(
            chartjs_generation_chain=chartjs_generation_chain,
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
        data = inputs[self.input_key]

        intermediate_steps: List = []

        generated_chartjs = self.chartjs_generation_chain.run(
            {"data": data}, callbacks=callbacks
        )

        try:
            js = generated_chartjs.replace("'",'"')
            js = json.dumps(json.loads(js))
            url = "https://quickchart.io/chart?c=" +urlencode({'c': js}, quote_via=quote_plus)
            print(url)
            result = run_sync(cl.Image(name="Graphique", url=url, display="inline").send())
            final_result = { "chart_url" : url}
        except:
            final_result = "Error while generating chart"
            logging.error("invalid json")
            logging.debug(generated_chartjs)

        intermediate_steps.append({"generate chart js specification ": generated_chartjs})



        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
