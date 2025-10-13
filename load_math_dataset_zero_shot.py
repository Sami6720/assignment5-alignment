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


with open("./cs336_alignment/prompts/r1_zero.prompt", "r") as f:

    prompt = f.readlines()
    prompt = '\n'.join(prompt)
    print(prompt)


question_answer_l = {"problems": [], "answers": []}
for data in ds:

    problem: str = data["problem"]
    answer: str = data["answer"]

    print(prompt.replace("{question}", problem))
    print("Answer: ", answer)
    problem = prompt.replace("{question}", problem)

    question_answer_l["problems"].append(problem)
    question_answer_l["answers"].append(answer)


with open(f"preprocessed_data/MATH/preprocessed_{split}.pkl", "wb") as f:
    pickle.dump(question_answer_l, f)


print(len(ds))

# with open(f"preprocessed_{split}.pkl", "rb") as f:
#     l = pickle.load(f)
#
#     print(l["problems"][:10])
#     print(l["answers"][:10])
