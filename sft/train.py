import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset, DataLoader
import pickle
from argparse import ArgumentParser
from utils import tokenize_prompt_and_outputs, get_response_log_probs, sft_microbatch_train_step, evaluate_vllm, init_vllm, load_policy_into_vllm_instance
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from functools import partial
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.optim import AdamW
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM




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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--job_name", type=str, default='default')
    parser.add_argument("--seed", type=int, default=0)


    #TODO: Set the seeds. Maybe the DataLoader seed also needs to be set?

    args = parser.parse_args()

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
        include_stop_str_in_output=True
    )



    assert args.batch_size % args.micro_batch_size == 0

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )

    datal = DataLoader(
        dataset=data,
        shuffle=True,
        batch_size=args.micro_batch_size,
        collate_fn=make_collate_fn(tokenizer)
    )


    for epoch in range(args.epochs):

        for i, d in enumerate(datal):


            input_ids = d['input_ids'].to(training_device)
            labels = d['labels'].to(training_device)
            response_mask = d['response_mask'].to(training_device)


            ret =  get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
            log_probs = ret["log_probs"]
            token_entropy = ret["token_entropy"]

            loss, meta_data = sft_microbatch_train_step(log_probs, response_mask, gradient_accumulation_steps=gradient_accumulation_steps)
            print(f"Iter: {i}, Loss: {loss}")

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()


            if (i+1) % args.eval_interval == 0:
                #TODO: Also save best eval model.
                load_policy_into_vllm_instance(model, llm)
                evals = evaluate_vllm(
                    llm, eval_data["problems"], eval_data["answers"], r1_zero_reward_fn,
                    sampling_params
                    )



            # evaluate_vllm(



    model.save_pretrained(save_directory=f"models/{args.job_name}/final/")
    tokenizer.save_pretrained(save_directory=f"models/{args.job_name}/final/")


