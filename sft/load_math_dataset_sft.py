from datasets import load_dataset
import pickle



# ds = load_dataset("~/links/scratch/huggingface/hub/hub/datasets--jhn9803--hendrycks-math-with-answers", split='train')
# import os
# os.environ["HF_DATASETS_OFFLINE"] = "1"

#NOTE: HF_DATASETS_OFFLINE=1 # This needs to be set.

split = 'train'

ds = load_dataset(
    "jhn9803/hendrycks-math-with-answers",
    split=split,
    # cache_dir="/lustre10/scratch/saminur/hf_cache/"
)



print(type(ds))

print(ds[0])

print(len(ds))


with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:

    main_prompt = f.readlines()
    main_prompt = '\n'.join(main_prompt)
    print(main_prompt)


question_answer_l = {"prompt": [], "output_strs": []}
for data in ds:

    prompt: str = data["problem"]
    thinkinging: str= data["solution"]
    answer: str = data["answer"]

    # print(prompt.replace("{question}", problem))
    # print("Answer: ", answer)
    prompt = main_prompt.replace("{question}", prompt)

    output_strs = thinkinging + "</think>"

    output_strs += "<answer>" + answer + "</answer>"


    print("Prompt_str: ", prompt)
    print("Output_str: ", output_strs)

    question_answer_l["prompt"].append(prompt)
    question_answer_l["output_strs"].append(output_strs)

import os
parent_dir = 'preprocessed_data/MATH/sft'
os.makedirs(parent_dir, exist_ok=True)
with open(f"{parent_dir}/preprocessed_{split}.pkl", "wb") as f:
    pickle.dump(question_answer_l, f)


print(len(ds))

# with open(f"preprocessed_{split}.pkl", "rb") as f:
#     l = pickle.load(f)
#
#     print(l["problems"][:10])
#     print(l["answers"][:10])
