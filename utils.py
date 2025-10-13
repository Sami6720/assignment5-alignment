from vllm import LLM, SamplingParams
from typing import List, Callable
from datasets import load_dataset


def evaluate_vllm(
        llm: LLM,
        prompts: List[str],
        ground_truths: List[str],
        reward_fn: Callable[[str, str], dict[str, float]],
        sampling_params: SamplingParams,
):

    # Generate texts from the prompts
    # The output is a list of RequestOutput objects that contain the prompt,
    # generated text, and other information
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs
    reward_and_response_per_prompt = []
    for output, ground_truth in zip(outputs, ground_truths):
        response = output.outputs[0].text
        reward = reward_fn(response, ground_truth)
        reward_and_response_per_prompt.append((reward, response))
        print(reward_and_response_per_prompt)
        break


    return reward_and_response_per_prompt
