import logging
import traceback


class ParserLogger:
    def __init__(self, logging_file="logs.log"):
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create file handler and set the logging level
        file_handler = logging.FileHandler(logging_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter and add it to the file handler
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    def log_info(self, message):
        self.logger.info(message)

    def log_test_result(self, input_text, predicitons, gt, score):
        self.logger.info(f"Input: {input_text}, Groundtruth: {gt}, score: {score}")

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error("An error occurred: %s", str(message))
        self.logger.error(traceback.format_exc())
