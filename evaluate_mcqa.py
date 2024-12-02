import json
import torch
import os
import sys

from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams


# Load the model and tokenizer
model_name = sys.argv[1]
data_path = Path("data")
BATCH_SIZE = 1
DEVICE = "cuda"
MAX_GENERATE_TOKENS = 1024
CHOICE_CANDIDATES = ['A','B','C','D','E','F','G','H']

llm = LLM(model=model_name, dtype="half", max_model_len=MAX_GENERATE_TOKENS, gpu_memory_utilization=0.95)

def get_allowed_token_ids(llm, allowed_tokens=['A','B','C','D']):
    return llm.llm_engine.tokenizer.tokenizer.convert_tokens_to_ids(allowed_tokens)

def ban_illegal_tokens(token_ids, logits, allowed_tokens):
    mask = torch.zeros_like(logits, dtype=torch.bool) # Mask for allowed tokens
    mask[allowed_tokens] = True
    
    logits = torch.where(mask, logits, torch.tensor(-float('inf')))
    return logits

def evaluate_model_mcqa(dataset: List[Dict[str, str]]):
    sampling_cot_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_GENERATE_TOKENS,
    )

    total_best_choices = []
    total_reasonings = []
    total_final_prompts = []

    batch_prompts = []
    batch_reasonings = []

    for example in tqdm(dataset):
        question = example["question"]
        choices = example["choices"]
        
        num_choices = len(choices)
        
        choice_candidates = CHOICE_CANDIDATES[:num_choices]
        
        allowed_token_ids = get_allowed_token_ids(llm, choice_candidates)

        sampling_final_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logits_processors=[lambda token_ids, logits: ban_illegal_tokens(token_ids, logits, allowed_token_ids)]
        )

        # Prepare the prompt with the given format
        choices_str = "\n".join(choices)
        prompt = f"다음 문제를 읽고 정답으로 가장 알맞은 것을 고르시오.\n### 질문: {question}\n### 선택지:\n{choices_str}\n### 정답:"
        batch_prompts.append(prompt)
        
        if len(batch_prompts) == BATCH_SIZE:
            outputs = llm.generate(batch_prompts, sampling_cot_params)
            batch_reasonings = [output.outputs[0].text for output in outputs]
            batch_final_prompts = [
                f"다음 문제를 읽고 정답으로 가장 알맞은 것을 고르시오.\n### 질문: {question}\n### 선택지:\n{choices_str}\n### 정답:{reasoning}\n### 정답:"
                for reasoning in batch_reasonings
            ]
            
            total_reasonings.extend(batch_reasonings)
            total_final_prompts.extend(batch_final_prompts)

            outputs = llm.generate(batch_final_prompts, sampling_final_params)
            
            batch_best_choices = [output.outputs[0].text for output in outputs]
            total_best_choices.extend(batch_best_choices)
            
            batch_prompts = []
            batch_final_prompts = []
        
    if len(batch_prompts) > 0:
        outputs = llm.generate(batch_prompts, sampling_cot_params)
        batch_reasonings = [output.outputs[0].text for output in outputs]
        batch_final_prompts = [
            f"다음 문제를 읽고 정답으로 가장 알맞은 것을 고르시오.\n### 질문: {question}\n### 선택지:\n{choices_str}\n### 정답:{reasoning}\n### 정답:"
            for reasoning in batch_reasonings
        ]

        total_reasonings.extend(batch_reasonings)
        total_final_prompts.extend(batch_final_prompts)
        
        outputs = llm.generate(batch_final_prompts, sampling_final_params)
        
        batch_best_choices = [output.outputs[0].text for output in outputs]
        total_best_choices.extend(batch_best_choices)

        batch_prompts = []
        batch_final_prompts = []

    results = []
    correct_count = 0  # Variable to count the number of correct answers

    for example, reasoning, final_prompt, best_choice in zip(dataset, total_reasonings, total_final_prompts, total_best_choices):
        question = example["question"]
        choices = example["choices"]
        correct_answer = example["answer"][0]

        # Check if the predicted answer matches the correct answer
        is_correct = best_choice == correct_answer
        if is_correct:
            correct_count += 1
        
        results.append({
            "question": question,
            "model_reasoning": reasoning,
            "final_prompt": final_prompt,
            "predicted_choice": best_choice,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        })
    
    # Calculate accuracy
    total_questions = len(dataset)
    accuracy = correct_count / total_questions * 100
    print(f"Total Accuracy: {accuracy:.2f}%")  # Print the final accuracy
    final_results = {"accuracy": round(accuracy, 2), "details": results}
    return final_results

for eval_mode in ["financial_market", "financial_accounting", "kmmlu_accounting", "kmmlu_accounting_hard"]:
    print(f"[EVALUATION]: {eval_mode}")

    # Load the dataset
    file_path = data_path / f"ko_eval_{eval_mode}.jsonl"

    with open(file_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, str]] = [json.loads(line) for line in f]

    # Run the evaluation
    results = evaluate_model_mcqa(dataset)

    model_name_for_save = model_name.split("/")[-1]

    os.makedirs(model_name_for_save, exist_ok=True)

    # Optionally, save to a file
    with open(Path(model_name_for_save) / f"{eval_mode}.results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
