from utils import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pickle
from vllm import LLM, SamplingParams


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



    llm = LLM(model="/home/saminur/links/scratch/huggingface/hub/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/")

    evals = evaluate_vllm(
        llm, data["problems"][0], data["answers"][0], r1_zero_reward_fn,
        SamplingParams()
        )









