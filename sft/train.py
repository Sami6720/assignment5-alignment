import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset, DataLoader
import pickle
from argparse import ArgumentParser
from utils import tokenize_prompt_and_outputs, get_response_log_probs, sft_microbatch_train_step, evaluate_vllm, init_vllm, load_policy_into_vllm_instance, log_evals_on_wandb, evalute_results
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, set_seed
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from functools import partial
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.optim import AdamW
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import wandb
import numpy as np
import random
import math



class SftDataset(Dataset):

    def __init__(self, path: str, ):

        with open(path, 'rb') as f:

            preprocessed_data = pickle.load(f)

        self.data = preprocessed_data
        self.len = len(preprocessed_data["prompt"])

    def __getitem__(self, index):

        return self.data["prompt"][index], self.data["output_strs"][index]

    def __len__(self,):

        return self.len


def make_collate_fn(tokenizer):
    def collate_fn(batch: list[dict]):
        prompts = []
        output_strs = []

        for b in batch:
            prompts.append(b[0])
            output_strs.append(b[1])

        return tokenize_prompt_and_outputs(prompts, output_strs, tokenizer)

    return collate_fn


if __name__ == '__main__':
    print("Starting training")

    if torch.cuda.device_count() > 1:
        training_device = 'cuda:0'
        eval_device = 'cuda:1'
    else:
        training_device = 'cuda:0'
        eval_device = 'cuda:0'

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--job_name", type=str, default='default')
    parser.add_argument("--seed", type=int, default=0)



    args = parser.parse_args()

    wandb.init(
        entity="doina-precup",
        project="cs-336-assignment-5",
        name=args.job_name,
        mode='online',
        save_code=True
    )
    wandb.define_metric("train_step") # the x‐axis for training 
    wandb.define_metric("eval_step") # the x‐axis for evaluation
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    set_seed(args.seed)
    vllm_set_random_seed(args.seed) # NOTE: Probably not needed.

    data = SftDataset('preprocessed_data/MATH/sft/preprocessed_train.pkl')
    with open("preprocessed_data/MATH/preprocessed_test.pkl", "rb") as f:
        eval_data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B",
                                                 torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                                 ).to(training_device)

    llm = init_vllm("Qwen/Qwen2.5-Math-1.5B", eval_device, args.seed)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed
    )


    assert args.batch_size % args.micro_batch_size == 0

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )


    def worker_init_fn(worker_id):
        # Recommended: derive from torch’s per-worker seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    datal = DataLoader(
        dataset=data,
        shuffle=True,
        batch_size=args.micro_batch_size,
        collate_fn=make_collate_fn(tokenizer),
        worker_init_fn=worker_init_fn,
        generator=g
    )


    global_training_step = 0
    global_eval_step = 0


    # --- compute how many optimizer steps you'll take ---
    MAX_VAL_LOGS = 15
    updates_per_epoch = math.ceil(len(datal) / gradient_accumulation_steps)
    total_updates = args.epochs * updates_per_epoch

    # choose up to 15 evenly spaced update indices: 0..total_updates-1
    eval_update_points = set(
        np.linspace(0, max(total_updates - 1, 0),
                    num=min(MAX_VAL_LOGS, total_updates),
                    dtype=int).tolist()
    )

    update_step = 0  # counts optimizer.step() calls
    for epoch in range(args.epochs):

        print("Len of datal ", len(datal))

        for i, d in enumerate(datal):


            input_ids = d['input_ids'].to(training_device)
            labels = d['labels'].to(training_device)
            response_mask = d['response_mask'].to(training_device)


            ret =  get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
            log_probs = ret["log_probs"]
            token_entropy = ret["token_entropy"]

            loss, meta_data = sft_microbatch_train_step(log_probs, response_mask, gradient_accumulation_steps=gradient_accumulation_steps)

            if global_training_step % 25 == 0:
                wandb.log({
                    "train_step": global_training_step,
                    "train/loss": loss,
                    "train/response_mean_token_entropy": (response_mask * token_entropy).mean().item()
                })

            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                # evenly spaced evals (at most 15 during training)
                if update_step in eval_update_points:
                    load_policy_into_vllm_instance(model, llm)
                    evals = evaluate_vllm(
                        llm, eval_data["problems"], eval_data["answers"],
                        r1_zero_reward_fn, sampling_params
                    )
                    eval_res, reward_0, format_1, reward_1 = log_evals_on_wandb(evals)
                    reward_0 = wandb.Table(columns=["prompt", "response"], data=reward_0)
                    format_1 = wandb.Table(columns=["prompt", "response"], data=format_1)
                    reward_1 = wandb.Table(columns=["prompt", "response"], data=reward_1)
                    wandb.log({"eval_step": global_eval_step, **eval_res, "reward_0_cases": reward_0, "reward_1_cases": reward_1, "format_1_cases": format_1})
                    global_eval_step += 1

                update_step += 1

            # evaluate_vllm(
            global_training_step += 1



    model.save_pretrained(save_directory=f"models/sft/{args.job_name}/final/")
    tokenizer.save_pretrained(save_directory=f"models/sft/{args.job_name}/final/")


    load_policy_into_vllm_instance(model, llm)
    evals = evaluate_vllm(
        llm, eval_data["problems"], eval_data["answers"], r1_zero_reward_fn,
        sampling_params
        )

    eval_res, reward_0, format_1, reward_1 = log_evals_on_wandb(evals)
    wandb.log({
        "eval_step": global_eval_step,
        **eval_res
    })
    report_path = evalute_results(evals, f"eval_outputs/sft/{args.job_name}")
    wandb.save(report_path)
