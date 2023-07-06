"""For communicating with OpenAI APIs, preparing the input and post-processing the outputs.

Classes:
    BASFMeasurementParser: For parsing text and communicating with OpenAI API using Langchain.

"""

import html
import os
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from logger import ParserLogger
from utils import build_dict_from_json_string

load_dotenv("env.env")


class BASFMeasurementParser:
    def __init__(
        self,
        prompt,
        prompt_id="000",
        chunk_size=3000,
        chunk_overlap=0,
        logger=None,
    ):
        """Parser main class.

        Handle communication with LLM, data pre-processing and data postprocessing.

        Args:
            prompt (str): The prompt that will be sent to the LMM.
            prompt_id (str, optional): Id of the prompt, each prompt has an ID,
            which is the name of the prompt's directory. Defaults to "000".
            chunk_size (int, optional): Big files are splitted into chunk,
            this is the chunk size for each chunk. Defaults to 3000.
            chunk_overlap (int, optional): Chunks overlapping size. Defaults to 0.
            logger (ParserLogger, optional): Logger to log info, warnings and errors
            for this process. Defaults to ParserLogger().
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.prompt = prompt
        self.BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.API_KEY = os.getenv("AZURE_OPENAI_KEY")
        self.DEPLOYMENT_NAME = "gpt-35-turbo"
        if not logger:
            self.logger = ParserLogger(f"logging/{prompt_id}_{chunk_size}_logs.log")
        else:
            self.logger = logger
        self.prompt_id = prompt_id

        try:
            self.logger.log_info("Initializing chat API..")
            self.chat_api = AzureChatOpenAI(
                openai_api_base=self.BASE_URL,
                openai_api_version="2023-03-15-preview",
                deployment_name=self.DEPLOYMENT_NAME,
                openai_api_key=self.API_KEY,
                openai_api_type="azure",
            )
            self.chain = LLMChain(llm=self.chat_api, prompt=self.prompt)
        except Exception as exception:
            log_message = f"Failed to initialize chat API, error message: {exception}"
            self.logger.log_error(log_message)

    def _load_txt_and_split(self, file_path):

        self.logger.log_info(f"Loading file from {file_path}.")
        text_loader = TextLoader(file_path, encoding="utf8")
        text = text_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        all_splits = text_splitter.split_documents(text)

        self.logger.log_info(
            f"File loaded and splitted into {len(all_splits)} chunks each with {self.chunk_size} size ."
        )
        return all_splits

    def parse_txt_by_chunks(
        self, file_path: str, output_format: str
    ) -> Tuple[dict, dict]:
        """Given a text file, it splits the text into chunks then feed it to LLM one by one.

        Chunks are sent to LLM one by one, output of the LLM is then post-processed to be
        a list of Json objects.


        Args:
            file_path (str): File to be parsed. File should contain text.
            output_format (str): The prompt for the output format.

        Returns:
            Tuple[dict, dict]: First item is the list of json objects,
            second item is predictions failed to be parsed as Json.
        """
        all_splits = self._load_txt_and_split(file_path)
        results_all_splits = []

        with open(
            f"parsing_results_{len(all_splits)}_chunks_prompt_{self.prompt_id}.txt",
            mode="w",
            encoding="utf8",
        ) as logs_file:
            for chunk in all_splits:

                result = self.parse_text(
                    input_text=chunk.page_content, output_format=output_format
                )
                results_all_splits.append(result)
                logs_file.write(html.unescape(result))
                logs_file.write("\n")

        self.logger.log_info("File Parsing has finished!")

        return self.post_process_predictions(results_all_splits)

    def post_process_predictions(
        self, predictions: list, return_df=True
    ) -> Tuple[dict, dict]:
        """Parse Predictions from LLM to follow JSON format.

        We parse the string to load it as JSON, some predictions fails to be parsed as JSON,
        they need further post_processing

        Args:
            predictions (list[str]): List of LLM outputs.
            return_df (bool, optional): IF true, returned predictions will be in a DataFrame. Default True.

        Returns:
            Tuple[dict, dict]: First item is the list of json objects,
            second item is predictions failed to be parsed as Json.
        """
        predictions_json = []
        false_json = []
        for prediction in predictions:
            try:
                predictions_json.extend(build_dict_from_json_string(prediction + "]"))
            except Exception as exception:
                self.logger.log_warning(
                    f"Failed to parse a prediction, Error: {exception}, Prediction: {prediction}"
                )
                false_json.append(prediction)
        if return_df:
            predictions_json = pd.DataFrame(predictions_json)
        return predictions_json, false_json

    def parse_text(self, input_text: str, output_format: str) -> list:
        """Parse a given text to extract Measurements.

        Args:
            input_text (str): Text to be parsed.
            output_format (str): Prompt string for the Output format.

        Returns:
           (list) : List of predictions.
        """

        try:
            self.logger.log_info(
                f"Sending {self.get_number_of_tokens(input_text)} tokens."
            )
            predictions = self.chain.run(
                input_text=input_text, output_format=output_format
            )

            return predictions
        except Exception as exception:
            # Handle the exception and generate an error message
            error_message = f"input text: {input_text}, Error: {exception}"
            print(error_message)
            self.logger.log_error(error_message)

            return error_message

    def get_number_of_tokens(self, prompt: str) -> int:
        """Calculate number of tokens of a given prompt text.

        Args:
            prompt (str): Prompt.

        Returns:
            int: Number of tokens.
        """
        return self.chat_api.get_num_tokens(prompt)
