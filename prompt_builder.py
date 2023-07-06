import html
import json
import os

from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from utils import format_dicts_to_string


class PromptBuilder:
    @classmethod
    def build_prompt_from_dir(cls, prompt_id):

        prompt = cls._load_prompt_from_json(prompt_id)
        examples = cls._load_examples_from_json(prompt_id)

        return (
            cls.build_few_shots_prompt(prompt["text"], examples),
            prompt,
            examples,
        )

    @classmethod
    def build_few_shots_prompt(cls, prompt, examples):

        system_message = (
            "You are a helpful assistant you extract measurements from research patent."
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_message
        )

        first_message = HumanMessagePromptTemplate.from_template(prompt)

        final_prompt = [system_message_prompt, first_message]

        for example in examples:
            final_prompt.append(
                HumanMessagePromptTemplate.from_template(
                    f"text: '''{example['text']}'''"
                )
            )
            final_prompt.append(
                AIMessagePromptTemplate.from_template(
                    format_dicts_to_string(example["measurements"])
                )
            )

        final_prompt.append(
            HumanMessagePromptTemplate.from_template(
                "\nOutput format:{output_format}:\n '''{input_text}'''"
            )
        )

        return ChatPromptTemplate.from_messages(final_prompt)

    @classmethod
    def _load_examples_from_json(cls, prompt_id: str) -> list[dict]:

        fie_path = f"Prompts/{prompt_id}/examples.json"
        if os.path.exists(fie_path):
            with open(fie_path, encoding="utf-8") as file:
                data = file.read()

            return json.loads(html.unescape(data))

        return []

    @classmethod
    def _load_prompt_from_json(cls, prompt_id):

        with open(f"Prompts/{prompt_id}/prompt.json", encoding="utf-8") as file:
            data = file.read()

        return json.loads(html.unescape(data))
