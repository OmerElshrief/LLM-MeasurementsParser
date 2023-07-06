import html
import json
from json.decoder import JSONDecodeError

import fuzzywuzzy.process as fuzz_process
from fuzzywuzzy import fuzz

from basf_measurement_parser import BASFMeasurementParser
from logger import ParserLogger
from prompt_builder import PromptBuilder
from utils import (
    build_dict_from_json_string,
    extract_numbers,
    write_json_objects_to_file,
)

logger = ParserLogger("Prompts/000/logs.log")


class PromptEvaluator:
    def evaluate_prompt_context(self, prompt_id):
        """
        Evaluates how the the LLM understand the provided examples within the prompt (in case of few-shots prompting),
        by using the example in the prompt as inputs then compare the predictions to the expected output.
        """
        (
            prompt,
            prompt_examples,
            prompt_instructions,
        ) = PromptBuilder.build_prompt_from_dir(prompt_id)
        # Building input and GT lists
        groud_truth = []
        inputs = []
        for exmaple in prompt_examples:
            groud_truth.append(exmaple["measurements"])
            inputs.append(exmaple["text"])

        # Defining a parser
        measurements_parser = BASFMeasurementParser(prompt)
        measurements_parser.build_chat_chain()

        output_format = prompt_instructions["output_format"]

        eval_results = []
        for input_text, gt in zip(inputs, groud_truth):

            predictions = html.unescape(
                measurements_parser.parse_text(input_text, output_format)
            )

            try:
                predictions_json = build_dict_from_json_string(predictions)
                eval_resul = self.evaluate_measurements_extraction(predictions_json, gt)
                eval_results.append(
                    {
                        "input": input_text,
                        "ground_truth": gt,
                        "predictions": predictions_json,
                        "score": eval_resul,
                    }
                )
            except JSONDecodeError as e:
                eval_results.append(
                    {
                        "input": input_text,
                        "ground_truth": gt,
                        "predictions": predictions,
                        "score": None,
                        "error": f"Output parsing error: {e.msg}",
                    }
                )
                logger.log_warning(
                    f"Context Evaluation score it not perfect, output: {predictions}, ground_truth: {gt}, score: {eval_resul}"
                )
                print(
                    f"Context Evaluation score it not perfect, output: {predictions}, ground_truth: {gt}, score: {eval_resul}"
                )

        self.analyze_evaluation_results(eval_results)
        # Let's add number of Tokens of the prompt
        eval_results[-1]["number_of_tokens"] = measurements_parser.get_number_of_tokens(
            prompt
        )
        write_json_objects_to_file(
            eval_results, f"prompts/{prompt_id}/context_evaluation_results.json"
        )

        return eval_results

    def evaluate_model_with_testset(self, prompt_id):

        with open("Data/test_set.json", encoding="utf-8") as file:
            data = file.read()

        test_data = json.loads(data)

        prompt, _, prompt_instructions = PromptBuilder.build_prompt_from_dir(prompt_id)

        measurements_parser = BASFMeasurementParser(prompt)
        measurements_parser.build_chat_chain()

        output_format = prompt_instructions["output_format"]
        eval_results = []

        for sample in test_data:

            input_text = sample["text"]
            gt = sample["measurements"]

            predictions = html.unescape(
                measurements_parser.parse_text(input_text, output_format)
            )

            try:
                predictions_json = build_dict_from_json_string(predictions)
                eval_resul = self.evaluate_measurements_extraction(predictions_json, gt)
                eval_results.append(
                    {
                        "input": input_text,
                        "ground_truth": gt,
                        "predictions": predictions_json,
                        "score": eval_resul,
                    }
                )
            except JSONDecodeError as e:
                eval_results.append(
                    {
                        "input": input_text,
                        "ground_truth": gt,
                        "predictions": predictions,
                        #             "score": eval_resul,
                        "error": f"Output parsing error: {e.msg}",
                    }
                )

        self.analyze_evaluation_results(eval_results)

        # Let's add number of Tokens of the prompt
        eval_results[-1]["number_of_tokens"] = measurements_parser.get_number_of_tokens(
            prompt
        )
        write_json_objects_to_file(
            eval_results, f"prompts/{prompt_id}/test_set_evaluation_results.json"
        )

        return eval_results

    def evaluate_measurements_extraction(self, gt, pred):

        evaluation_results = []
        true_positives = 0
        missing_predictions = 0

        for pred_item in pred:

            measurement = pred_item["measurement"]
            matching_gt_item, similarity = fuzz_process.extractOne(
                measurement, [gt_item["measurement"] for gt_item in gt]
            )
            pred_unit = pred_item["unit"].replace("-", " ").replace("to", " ")
            pred_value = extract_numbers(pred_item["value"])
            #         print(matching_gt_item, pred_item, similarity)

            if matching_gt_item:
                matching_gt_items = [
                    gt_item
                    for gt_item in gt
                    if gt_item["measurement"] == matching_gt_item
                ]

                for matching_gt_item in matching_gt_items:

                    gt_unit = (
                        matching_gt_item["unit"].replace("-", " ").replace("to", " ")
                    )
                    gt_value = extract_numbers(matching_gt_item["value"])

                    unit_match = fuzz.ratio(gt_unit, pred_unit) > 0.8
                    value_match = gt_value == pred_value

                    if similarity >= 70 and unit_match and value_match:
                        evaluation_results.append(
                            {"measurement": measurement, "match": True}
                        )
                        true_positives += 1
            else:
                missing_predictions += 1

        false_predicted_samples = len(gt) - true_positives

        accuracy = true_positives / len(pred)

        evaluation_results.append(
            {
                "true_predictions": true_positives,
                "accuracy": accuracy,
                "missing_predictions": missing_predictions,
                "false_predicted_samples": false_predicted_samples,
            }
        )

        return evaluation_results

    def analyze_evaluation_results(self, eval_results):

        total_overall_accuracy = 0
        total_true_predictions = 0
        total_miss_predictions = 0
        total_false_predictions = 0

        for result in eval_results:
            if result["score"]:
                total_overall_accuracy += result["score"][-1]["accuracy"]
                total_true_predictions += result["score"][-1]["true_predictions"]
                total_miss_predictions += result["score"][-1]["missing_predictions"]
                total_false_predictions += result["score"][-1][
                    "false_predicted_samples"
                ]
            else:
                total_overall_accuracy += 0
                total_true_predictions += 0
                total_miss_predictions += 0
                total_false_predictions += 1

        total_overall_accuracy /= len(eval_results)

        total_result = {}
        total_result["total_overall_accuracy"] = total_overall_accuracy
        total_result["total_true_predictions"] = total_true_predictions
        total_result["total_miss_predictions"] = total_miss_predictions
        total_result["total_false_predictions"] = total_false_predictions

        eval_results.append(total_result)
