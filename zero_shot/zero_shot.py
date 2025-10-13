from utils import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pickle
from vllm import LLM, SamplingParams
import pprint


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
    

    stat_correct_all = 0
    stat_format_reward_1_answer_reward_0 = 0
    stat_format_reward_0_answer_reward_0 = 0

    format_reward_0_cases = []
    format_reward_1_answer_reward_0 = []
    reward_1_cases = []
    for eval in evals:
        reward = eval[0]

        format_reward = reward["format_reward"]
        answer_reward = reward["answer_reward"]
        reward = reward["reward"]


        if reward == 1 and format_reward == 1 and answer_reward == 1:
            stat_correct_all += 1

            if len(reward_1_cases) <= 20:
                reward_1_cases.append(eval)


        if format_reward == 1 and answer_reward == 0:
            stat_format_reward_1_answer_reward_0 += 1

            if len(format_reward_1_answer_reward_0) <= 10:
                format_reward_1_answer_reward_0.append(eval)


        if answer_reward == 0 and format_reward == 0:
            stat_format_reward_0_answer_reward_0 += 1

            if len(format_reward_0_cases) <= 10:
                format_reward_0_cases.append(eval)





    # collect everything in one dict for easy saving and display
    results = {
        "stat_correct_all": stat_correct_all,
        "stat_format_reward_1_answer_reward_0": stat_format_reward_1_answer_reward_0,
        "stat_format_reward_0_answer_reward_0": stat_format_reward_0_answer_reward_0,
        "format_reward_0_cases": format_reward_0_cases,
        "format_reward_1_answer_reward_0": format_reward_1_answer_reward_0,
    }

    # persist to pickle
    with open("eval_stats.pkl", "wb") as f:
        pickle.dump(results, f)


    results = {
        "stat_correct_all": stat_correct_all,
        "stat_format_reward_1_answer_reward_0": stat_format_reward_1_answer_reward_0,
        "stat_format_reward_0_answer_reward_0": stat_format_reward_0_answer_reward_0,
        "format_reward_0_cases": [('prompt: ' + eval[1], 'response: ' + eval[2]) for eval in format_reward_0_cases],
        "format_reward_1_answer_reward_0": [('prompt: ' + eval[1], 'response: ' + eval[2]) for eval in format_reward_1_answer_reward_0],
        "reward_1_cases": [('prompt: ' + eval[1], 'response: ' + eval[2]) for eval in reward_1_cases],
    }

    # print nicely
    print("\n=== Evaluation Summary ===")
    pprint.pprint(results)
