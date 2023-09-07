from typing import Dict, Any, Optional

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage


class MyMemory(ConversationBufferMemory):
    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        input_msg, output_msg = self._get_input_output(inputs, outputs)
        if type(input_msg) is list:
            for msg in input_msg:
                if type(msg) is HumanMessage:
                    self.chat_memory.add_user_message(str(msg.content))
        self.chat_memory.add_ai_message(str(output_msg))
