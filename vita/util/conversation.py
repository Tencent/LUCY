import re
import dataclasses
from enum import Enum, auto
from typing import List

class Version(Enum):
    QWEN2 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""                                          

    system: str
    roles: List[str]
    messages: List[List[str]]
    sep: str = "###"
    version: str = "Unknown"
    skip_next: bool = False
    func_pattern: str = ""

    def get_prompt(self):
        rets, is_target = self.get_prompt_split_by_target()
        ret = "".join(rets)
        return ret

    def get_prompt_split_by_target(self):
        # return prompts as:
        # [
        #   "<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n,
        #   "{message}<|im_end|>",
        #   "\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n",
        #   "{message}<|im_end|>",
        #   "\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n",
        #   "{message}<|im_end|>",
        #   ...
        #   "\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n",
        #   "{message}<|im_end|>",
        # ]
        messages = self.messages
        wrap_qa = lambda msg: f"<|im_start|>{msg}<|im_end|>\n"
        rets = [wrap_qa(f"system\n{self.system}")]
        is_target = [False]
        # assert len(messages) % 2 == 0, len(messages)
        for i, (role, message) in enumerate(messages):
            assert messages, f"message: {message} is empty"
            if type(message) is tuple:
                message, _, _ = message
            
            head = f"<|im_start|>{role}\n"
            body = f"{message}<|im_end|>"
            tail = f"\n"
            if role == "assistant":
                rets[-1] += head
                rets.append(body)
                rets.append(tail)
                is_target.extend([True, False])
            elif role == "user" or role == "observation":
                rets[-1] += f"{head}{body}{tail}"
            else:
                raise ValueError(f"{role} not implemented")
        if role == "assistant":
            rets, is_target = rets[:-1], is_target[:-1]
        return rets, is_target

    def has_function_call(self, message):
        has_func_call = re.search(self.func_pattern, message) is not None
        return has_func_call
    
    def remove_function_call(self, message):
        has_func_call = self.has_function_call(message)
        if has_func_call:
            ret, = re.match(f"(.*)(?={self.func_pattern})", message.strip()).groups()
            ret = f"{ret}"
            return ret
        return message

    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep=self.sep,
            version=self.version,
            func_pattern=self.func_pattern,
        )

conv_qwen2 = Conversation(
    system="You are an AI robot and your name is VITA. \n"
        "- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n"
        "- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n"
        "- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.",
    roles=("user", "assistant", "observation"),
    version=Version.QWEN2,
    messages=(),
    sep="<|im_start|>",
    func_pattern="<function=.*</function>",
)
