import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE_URL")

from abc import ABC, abstractmethod


class ChatLLM(ABC):
    @abstractmethod
    def invoke(self, messages: list[dict[str, str]]) -> str:
        pass

    @abstractmethod
    def rag_agent_route_embedding(self, text: str) -> list:
        pass


class OpenAIChatLLM(ChatLLM):
    def __init__(self, model: str):
        self.model = model

    def invoke(self, messages: list[dict[str, str]]) -> str:
        """
        Invoke the OpenAI LLM with the given messages.

        Parameters
        ----------
        messages : list[dict[str, str]]
            The messages to send to the LLM.
            e.g. [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}]

        Returns
        -------
        str
            The response from the LLM.
            e.g. "Hello! How can I help you today?"
        """
        openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        response = openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

    def rag_agent_route_embedding(self, text: str) -> list:
        openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        response = openai_client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding