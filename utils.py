import ast
import json
import re
import string


def write_json_objects_to_file(json_objects, file_path):
    """
    Writes a list of JSON objects to a file.

    Args:
        json_objects (list): List of JSON objects.
        file_path (str): File path to write the JSON objects to.
    """
    with open(file_path, "w") as file:
        json.dump(json_objects, file)


def format_dicts_to_string(data_list):
    formatted_strings = []

    for data in data_list:
        formatted_string = '''"measurement": "{measurement}", "unit": "{unit}", "value": "{value}"'''.format(
            **data
        )
        formatted_strings.append(formatted_string)

    result = ",\n".join(formatted_strings)
    return result


def extract_numbers(string):

    string = string.replace("-", " ")
    pattern = r"-?\d+(?:\.\d+)?"
    float_numbers = re.findall(pattern, string)
    float_numbers = [float(num) for num in float_numbers]

    return float_numbers


def build_dict_from_json_string(json_str_list: list) -> list:
    """Building structured dict form the predictions.

    Args:
        json_str_list (list): list of predictions.

    Returns:
        list: List of json objects.
    """
    try:
        fixed_json = json.loads(json_str_list)
        for prediction in fixed_json:
            prediction = {
                key.strip(): value.strip() for key, value in prediction.items()
            }
        return fixed_json
    except json.JSONDecodeError as e:
        # Remove the surrounding square brackets and newlines
        clean_string = json_str_list.replace("'", '"')

        # Safely evaluate the string as a list of dictionaries
        dictionary_list = ast.literal_eval(clean_string)
        for prediction in dictionary_list:
            prediction = {
                key.strip(): value.strip() for key, value in prediction.items()
            }
        return dictionary_list
