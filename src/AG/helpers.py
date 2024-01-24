from typing import List, Tuple

import io
import contextlib
import re

import threading
from tqdm import tqdm


def extract_assistant_completion(completion: str) -> str:
    """
    Extracts the portion of the completion that comes after the [/INST] tag.
    """
    end_tag = "[/INST]"
    end_idx = completion.find(end_tag)

    if end_idx != -1:
        # Extracting the portion after the end tag
        post_inst_content = completion[end_idx + len(end_tag):]
        return post_inst_content.strip()  # Strips leading/trailing whitespace
    else:
        return ""  # Returns an empty string if the end tag is not found


def merge_datasets(
    original, 
    updates, 
    problem_id_key='problem_id',
) -> List[dict]:
    """Merge datasets."""
    original_list = [item for item in original]
    update_ids = {item[problem_id_key]: item for item in updates}
    for i, item in enumerate(original_list):
        if item[problem_id_key] in update_ids:
            original_list[i].update(update_ids[item[problem_id_key]])
    for item in updates:
        if item[problem_id_key] not in [original_item[problem_id_key] for original_item in original_list]:
            original_list.append(item)
    return original_list


def format_llama_message(
        system_message: str, 
        human_message: str, 
        assistant_message: str
    ) -> str:
    """Format a message to fit Llama format. Based on:
        https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k
        https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html).
    """
    return """<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{human_message} [/INST] {assistant_message}</s>""".format(system_message=system_message, 
                                                      human_message=human_message, 
                                                      assistant_message=assistant_message)


def get_llama_messages(
    problem_description: str,
    initial_solution: str,
    assistant_response: str,
    ) -> str:
    """Get messages."""

    system_message = "You are an expert computer science researcher and programmer, especially skilled at debugging algorithms."

    human_message = f"""I have to solve the following problem:

{problem_description}

Here is my initial solution to the problem:
```python
{initial_solution}
```
Insert print statements in the program that will help me debug and improve the program."""
    assistant_message = f"""{assistant_response}"""
    return system_message, human_message, assistant_message


def tokenize_fn(
    text,
    tokenizer,
    max_length: int = 512,
    padding: str = "longest",
    return_tensors=None,
    truncation: bool = True,
    ignore_index: bool = False,
    ):
    """Simple LLama tokenize function."""
    result = tokenizer(
        text,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )
    if ignore_index:
        raise f"ignore_index not implemented; need to add -100 to labels"
    else:
        result["labels"] = result["input_ids"].copy()
    return result


def evaluate_solutions(
        solution,
        input_output_pairs,
) -> Tuple[float, List[str], List[str]]:
    """Evaluate solutions."""
    results, print_outputs, error_messages = [], [], []   
    try:
        exec(solution, globals())
        for input, output in zip(input_output_pairs['inputs'], input_output_pairs['outputs']):
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                try:
                    result = solution_algorithm(input)
                    results.append(compare_outputs(result, output))
                except Exception as e:
                    results.append(False)
                    error_messages.append(str(e))
                print_outputs.append(buf.getvalue())
    except Exception as e:
        error_messages.append(f"Error in executing solution: {e}")
        print_outputs.append("")

    accuracy = sum(1 for r in results if r == True) / len(results) if results else 0.0
    deterministic_accuracy = 1 if sum(results) == len(results) else 0
    return accuracy, deterministic_accuracy,print_outputs, error_messages


def compare_outputs(
    actual: Tuple[int, ...], 
    expected: str,
) -> bool:
    """Compare outputs."""
    expected_normalized = expected.strip().split()
    if isinstance(actual, (tuple, list)):
        actual_normalized = [str(item) for item in actual]
    else:
        actual_normalized = [str(actual)]

    return actual_normalized == expected_normalized

def remove_comments(
    code: str,
) -> str:
    """Remove comments from code."""
    code_without_comments = re.sub(r'#.*', '', code)
    return code_without_comments