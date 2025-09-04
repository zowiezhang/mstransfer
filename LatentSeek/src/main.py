from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from data import get_dataset
from tqdm import tqdm
from rewards.reward import RewardModel
from ori_generation import original_generation
from opt_generation import optimized_generation
import os
from extract_judge_answer import extract_answer, extract_true_answer, judge_answer
import argparse
import numpy as np
import random
import ipdb

from utils.load_json import load_json
from utils.save_json import save_json
from prompts.prompts import SYSTEM_PROMPT, PREFERENCE_PROMPTS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="Dataset to evaluate")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--pref_dim", type=str, default = 'creative', help="the preference dimension")
    
    parser.add_argument("--output_dir", type=str, default = '/home/nicolas/desktop/latentalin/LatentSeek/responses', help="Path to the output directory")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=1319, help="End index of the data to evaluate")

    # prompt
    parser.add_argument("--solver_prompt_idx", type=int, default=0, help="Index of the solver prompt")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # optimization args
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_num_steps", type=int, default=10, help="Number of optimization iterations")
    # parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of generated tokens")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of generated tokens")
    parser.add_argument("--device", type=str, default=None)

    # format reward
    parser.add_argument("--rule_format_string", type=str, default=None, help="the answer format that should follow")

    # reward model
    parser.add_argument("--reward_threshold", type=float, default=-1, help="Threshold for reward to stop optimization")

    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    return parser.parse_args()


def set_seed(seed):
    '''
    Set random seed for reproducibility

    Args:
        seed: random seed
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# evaluate function 
def main(args):
    '''
    Evaluate model

    Args:
        dataset: dataset to evaluate
        sample_num: number of samples to evaluate

    Returns:
        original_accuracy: original generation accuracy
        optimized_accuracy: optimized generation accuracy
    '''

    # if args.rule_format_string == "boxed":
    #     rule_format_string = r'\\boxed{(.*)}'
    # else:
    #     if args.rule_format_string:
    #         raise ValueError("Unknown format")
    #     rule_format_string = None
    
    if args.seed:
        set_seed(args.seed)
    
    # set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load reward model
    reward_model = RewardModel(
            model=model, 
            tokenizer=tokenizer, 
            device=device,
            # data_name=args.dataset,
            # rule_format_string=rule_format_string
            )

    # load dataset
    # dataset = load_json('/home/shiweiye/zhaowei/LatentSeek/datas/personal_preference_eval_preference_data.json')
    dataset = get_dataset(args.dataset, 
                          tokenizer=tokenizer,
                          prompt_idx=args.solver_prompt_idx,
                          args = args, )
    # dict_keys(['question', 'answer', 'formatted'])
    print(f"Example: {dataset[0]}")
    
    original_answers = []
    optimized_answers = []
    
    # pref_dim = 'concise'
    # pref_input = f"Your answer should be as {pref_dim} as possible."

    original_correct = 0
    optimized_correct = 0
    total = 0
    update_count = 0
    original_length = 0
    optimized_length = 0
    fitten_length = 0
    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.dataset.split("/")[-1]

    output_dir = f"{args.output_dir}/{model_name}-{data_name}-k{args.k}-lr{args.lr}-SolIdx{args.solver_prompt_idx}"

    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = min(args.end_data_idx, len(dataset))
  
    if args.resume:
        print(f"Resume from {output_dir}")
        # load logistics
        logistics = torch.load(f"{output_dir}/logistics.pt")
        start_data_idx = logistics["start_idx"]
        original_correct = logistics["original_correct"]
        optimized_correct = logistics["optimized_correct"]
        total = logistics["total"]
        update_count = logistics["update_count"]
        original_length = logistics["original_length"]
        optimized_length = logistics["optimized_length"]
        fitten_length = logistics["fitten_length"]
    
    print(f"Start to evaluate {args.dataset} from {start_data_idx} to {end_data_idx}...")

    data_idx_list = range(start_data_idx, end_data_idx)
    # data_idx_list = range(0, 20)
    # data_idx_list = range(0, 10)
    for i in tqdm(data_idx_list):
        example = dataset[i]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(f"{output_dir}/test"):
            os.makedirs(f"{output_dir}/test")


        print(f"Question: {example['question']}")
        
        original_output, hidden_states_list, input_ids = original_generation(
                input_text=example["formatted"],
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                device=device,)

        # ipdb.set_trace()
        
        optimized_output, reward_history, new_original_length, new_optimized_length, new_update_length = optimized_generation(
                reward_model=reward_model,
                model=model,
                tokenizer=tokenizer,
                device=device,
                question=example["question"],
                pref_input=PREFERENCE_PROMPTS[args.pref_dim],
                input_text=example["formatted"],
                original_answer=original_output,
                original_hidden_states_list=hidden_states_list, 
                input_ids=input_ids,
                max_num_steps=args.max_num_steps,
                lr=args.lr,
                max_new_tokens = args.max_new_tokens,
                grad_clip=args.grad_clip,
                k=args.k,
                reward_threshold=args.reward_threshold,
        )
        
        # ipdb.set_trace()

        update_count += (len(reward_history) - 1)   
        
        # original_answer = original_output
        optimized_answer = optimized_output
        
        # original_answers.append({'question': example["question"], 
        #                        'preference': args.pref_dim,
        #                        'responses': original_output})
        optimized_answers.append({'question': example["question"], 
                               'preference': args.pref_dim,
                               'responses': optimized_answer})
        
        original_length += new_original_length
        optimized_length += new_optimized_length
        fitten_length += (new_optimized_length - new_update_length) if len(reward_history) > 1 else 0

        

        total += 1

        # save logistics
        # save original correct, optimized correct, total, update count
        # torch.save({
        #     "original_correct": original_correct,
        #     "optimized_correct": optimized_correct,
        #     "total": total,
        #     "start_idx": i+1,
        #     "update_count": update_count,
        #     "original_length": original_length,
        #     "optimized_length": optimized_length,
        #     "fitten_length": fitten_length
        # }, f"{output_dir}/logistics.pt")
        

    rst_output_dir = '/home/nicolas/desktop/latentalin/LatentSeek/responses'
    # save_json(original_answers, f"{rst_output_dir}/original_answers.json")
    save_json(optimized_answers, f"{rst_output_dir}/optimized_answers.json")

    # print(f"Original accuracy: {original_correct / total:.4f}")
    print(f"Optimized accuracy: {optimized_correct / total:.4f}")
    print(f"Average update length: {update_count / total:.4f}")
    # print(f"Average original length: {original_length / total:.4f}")
    print(f"Average optimized length: {optimized_length / total:.4f}")
    print(f"Average fitten length: {fitten_length / total:.4f}")       


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)


