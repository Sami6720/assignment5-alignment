from utils import evaluate_vllm, evalute_results
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pickle
from vllm import LLM, SamplingParams
import pprint
import os
from datetime import datetime


sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True
)

if __name__ == "__main__":

    with open("preprocessed_data/MATH/preprocessed_test.pkl", "rb") as f:
        data = pickle.load(f)



    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

    evals = evaluate_vllm(
        llm, data["problems"], data["answers"], r1_zero_reward_fn,
        sampling_params
        )

    evalute_results(evals)
