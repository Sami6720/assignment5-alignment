from vllm import LLM, SamplingParams
from typing import List, Callable
from datasets import load_dataset
import os
import pprint
import pickle
from datetime import datetime
from transformers import PreTrainedTokenizer
import torch


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
    for output, ground_truth, prompt in zip(outputs, ground_truths, prompts):
        response = output.outputs[0].text
        reward = reward_fn(response, ground_truth)
        reward_and_response_per_prompt.append((reward, prompt, response, ground_truth))

    return reward_and_response_per_prompt


def evalute_results(evals):

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

    results = {
        "stat_correct_all": stat_correct_all,
        "stat_format_reward_1_answer_reward_0": stat_format_reward_1_answer_reward_0,
        "stat_format_reward_0_answer_reward_0": stat_format_reward_0_answer_reward_0,
        "format_reward_0_cases": [
            {"prompt": e[1], "response": e[2], "ground_truth": e[3]} for e in format_reward_0_cases
        ],
        "format_reward_1_answer_reward_0": [
            {"prompt": e[1], "response": e[2], "ground_truth": e[3]} for e in format_reward_1_answer_reward_0
        ],
        "reward_1_cases": [
            {"prompt": e[1], "response": e[2], "ground_truth": e[3]} for e in reward_1_cases
        ],
    }

    out_dir = "eval_outputs/zero_shot/"
    os.makedirs(out_dir, exist_ok=True)

    # --- 1) Persist with pickle ---
    with open(os.path.join(out_dir, "eval_stats.pkl"), "wb") as f:
        pickle.dump(results, f)

    # --- 2) Markdown report with ground truth included ---
    report_path = os.path.join(out_dir, "eval_report.md")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def section(title: str) -> str:
        return f"\n## {title}\n\n"

    def render_cases(title: str, cases: list[dict]) -> str:
        lines = [section(f"{title} (N={len(cases)})")]
        if not cases:
            lines.append("_No cases._\n")
            return "".join(lines)
        for i, c in enumerate(cases, 1):
            prompt = (c.get("prompt") or "").rstrip()
            response = (c.get("response") or "").rstrip()
            gt = (c.get("ground_truth") or "").rstrip()
            lines.append(f"### Case {i}\n")
            lines.append("**Prompt**\n\n```text\n" + prompt + "\n```\n\n")
            lines.append("**Response**\n\n```text\n" + response + "\n```\n\n")
            lines.append("**Ground Truth**\n\n```text\n" + gt + "\n```\n\n")
            lines.append("---\n\n")
        return "".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Report\n\n_Generated: {ts}_\n\n")
        f.write("### Summary Stats\n\n")
        f.write(f"- **stat_correct_all**: {results['stat_correct_all']}\n")
        f.write(
            f"- **stat_format_reward_1_answer_reward_0**: {results['stat_format_reward_1_answer_reward_0']}\n")
        f.write(
            f"- **stat_format_reward_0_answer_reward_0**: {results['stat_format_reward_0_answer_reward_0']}\n")
        f.write(render_cases(
            "Cases: format_reward == 1 & answer_reward == 0",
            results["format_reward_1_answer_reward_0"]
        ))
        f.write(render_cases(
            "Cases: format_reward == 0 & answer_reward == 0",
            results["format_reward_0_cases"]
        ))
        f.write(render_cases(
            "Cases: reward == 1 (correct)",
            results["reward_1_cases"]
        ))

    print("✅ Saved:")
    print(f" - Pickle: {os.path.join(out_dir, 'eval_stats.pkl')}")
    print(f" - Report: {report_path}")


def tokenize_prompt_and_outputs(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:


    prompt_encoded = tokenizer.batch_encode_plus(prompt_strs)["data"]
    output_encoded = tokenizer.batch_encode_plus(output_strs)["data"]

    tokens_all = []
    max_o_p_len = float('-inf')
    len_p = []
    len_o = []
    for i in range(len(output_encoded)):

        p = prompt_encoded[i]
        len_p.append(len(p))

        o = prompt_encoded[i]
        len_o.append(len(o))
        p_o= p + o

        max_o_p_len = max(len(p_o), max_o_p_len)

        tokens_all.append(p)


    max_o_p_len = int(max_o_p_len)
    mask = torch.ones(size=(len(prompt_strs), max_o_p_len))

    for i, m in enumerate(mask):

        m[:len_p[i]] = 0
        m[len_p[i] + len_o[i] + 1:] = 0

    tokens_all: torch.Tensor = tokenizer.pad(tokens_all).convert_to_tensors()


    print(tokens_all.shape)


    return {
        "input_ids": tokens_all[:, :-1],
        "labels": tokens_all[:, 1:],
        "response_mask": mask[:, 1:],
    }
