from typing import Dict, Any, Optional, List
from langchain.memory import ConversationEntityMemory
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

class ExtendedConversationEntityMemory(ConversationEntityMemory):
    extra_variables:List[str] = []
    memory_key = "history"
    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return buffer with history and extra variables"""
        d = super().load_memory_variables(inputs)
        d.update({k:inputs.get(k) for k in self.extra_variables})
        return d