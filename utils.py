from vllm import LLM, SamplingParams
from typing import List, Callable
from datasets import load_dataset
import os
import pprint
import pickle
from datetime import datetime
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
import wandb


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

def log_evals_on_wandb(evals):

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

            if len(reward_1_cases) <= 10:
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
        "eval/correct": stat_correct_all / len(evals) * 100,
        "eval/stat_format_reward_1_answer_reward_0": stat_format_reward_1_answer_reward_0 / len(evals) * 100,
        "eval/stat_format_reward_0_answer_reward_0": stat_format_reward_0_answer_reward_0 / len(evals) * 100,
    }

    return ( results, [(eval[1], eval[2]) for eval in format_reward_0_cases], [(eval[1],  eval[2]) for eval in format_reward_1_answer_reward_0], [(eval[1],  eval[2]) for eval in reward_1_cases])



def evalute_results(evals, out_dir="eval_outputs/zero_shot/"):

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



    stat_correct_all /= len(evals) * 100
    stat_format_reward_1_answer_reward_0 /= len(evals) * 100
    stat_format_reward_0_answer_reward_0 /= len(evals) * 100

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

    # out_dir = "eval_outputs/zero_shot/"
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


    return report_path


def tokenize_prompt_and_outputs(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:



    resp_token = "<|resp|>"

    prompt_strs_plus_output_strs = []

    for p, o in zip(prompt_strs, output_strs):
        prompt_strs_plus_output_strs.append(p + resp_token + o)

    tokenizer.add_special_tokens({"additional_special_tokens": [resp_token]})
    resp_token_id = tokenizer.convert_tokens_to_ids(resp_token)

    all_tokens = tokenizer.batch_encode_plus(prompt_strs_plus_output_strs, add_special_tokens=False, return_tensors='pt', padding=True)["input_ids"]

    mask = all_tokens != resp_token_id
    tokens = all_tokens[mask].reshape(all_tokens.shape[0], -1)


    # Mask over input prompts, remove the resp_id column, mask over the padds
    resp_token_mask  = all_tokens == resp_token_id
    # cumulative summation returns the same shape before and after
    # everything before the resp_token_id is the input prompt
    input_token_mask = torch.cumsum(resp_token_mask, dim=-1)

    # padding mask
    pad_token_id = tokenizer.eos_token_id # Eos tokens are used for padding.
    pad_token_mask = all_tokens != pad_token_id
    # The shape of input_token_mask and pad_token_mask are the same.
    all_token_mask  = pad_token_mask & input_token_mask

    response_mask = all_token_mask[mask].reshape(all_token_mask.shape[0], -1)


    return {
        "input_ids": tokens[:, :-1],
        "labels": tokens[:, 1:],
        "response_mask": response_mask[:, 1:],
    }


def per_token_entropy(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=-1)
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    mult = logits - logsumexp
    return -1 * (probs * mult).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool):

    ret = {}


    print("input_ids.shape ", input_ids.shape)


    logits = model(input_ids).logits # Shape B T V

    if return_token_entropy:
        ret["token_entropy"] = per_token_entropy(logits) # Shape B T

    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True) # Shape B T 1
    log_prob = logits - logsumexp # B T V

    ret["log_probs"] = torch.gather(input=log_prob, index=labels.unsqueeze(-1), dim=-1).squeeze() # B T
    # index determines the shape of the output.
    # As you construct the output, you loop over the dims of index
    # index and input must have the same number of dims. This is just a torch req.

    return ret



def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0
    ):


    masked_tensor = tensor * mask

    masked_tensor = masked_tensor.sum(dim=dim)

    return masked_tensor/normalize_constant



def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,):

    """
    Lets say you have main batch of size $B$. 
    Let say you have `gradient_accumulation_step` be $k$.

    Actual $loss = \frac{1}{B}SE_{B}$ per microbatch.

    So the effective batch size would be $\frac{B}{k}$. Let this value be $b$. In other wods, $b = \frac{B}{k}$

    $$total\_loss = \frac{1}{B}\sum^{k}_{i=1}SE_{b}$$


    For each `loss.backward()` the loss would be over 
    $\frac{1}{b}SE_b$

    For $b$ such updates the 

    $$total\_acheived\_loss = \frac{1}{b}\sum^{k}_{i=1}SE_{b}$$
    $\frac{1}{b}$ appears in all the terms so its moved out of the summation.

    Since $B = b \times k$, we need to divide the total_achieved loss by $k$. This could be distributed to per microbatch $k$ microbatch of size $b$. 
    """

    B, T = policy_log_probs.shape
    #
    summed_log_nll = -1 * masked_normalize(policy_log_probs, response_mask, -1, normalize_constant)
    # # loss.sum().backward()
    # #
    # # return loss.sum(), {"gradient_accumulation_steps": gradient_accumulation_steps + 1}
    #
    #
    # # Per-example masked sum over response tokens, then divide by the provided constant
    # per_example_loss = -(policy_log_probs * response_mask).sum(dim=-1) / float(normalize_constant)  # (B,)

    # What we backprop through: batch **mean**, scaled for grad accumulation. You weren't meaning!
    loss = summed_log_nll.mean()/ gradient_accumulation_steps  # scalar

    loss.backward()
    return loss.item(), {"gradient_accumulation_steps": gradient_accumulation_steps}



from unittest.mock import patch

import torch
from transformers import PreTrainedModel
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    """
    Start the inference process; use vLLM to hold a model on a GPU separate from the policy.

    Applies two patches (following TRL) to:
      (1) force world size to 1 so we can place the vLLM model on the desired device, and
      (2) bypass a profiling check not designed for this setting.
    """
    vllm_set_random_seed(seed)

    # Monkey patches adapted from:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )

    with world_size_patch, profiling_patch:
        #TODO: Maybe I need to add SamplingParams here.
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """
    Load a Hugging Face `policy` state_dict into an existing vLLM `llm` instance.

    Based on:
    https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670
    """
    state_dict = policy.state_dict()
    llm_model = (
        llm.llm_engine.model_executor.driver_worker.model_runner.model  # type: ignore[attr-defined]
    )
    llm_model.load_weights(state_dict.items())

