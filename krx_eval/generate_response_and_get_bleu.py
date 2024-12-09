import json
import sys
import os

from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams
from konlpy.tag import Okt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf


MAX_GENERATE_TOKENS = 2048

okt = Okt()

sampling_cot_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_GENERATE_TOKENS,
)

def tokenize_korean(sentence):
    """
    Tokenize a Korean sentence using Okt.
    
    Parameters:
    - sentence (str): A Korean sentence.
    
    Returns:
    - list: A list of tokens.
    """
    return okt.morphs(sentence)

def calculate_bleu(reference_list, candidate_list):
    """
    Calculate BLEU score for Korean sentences.
    
    Parameters:
    - reference_list (list): A list of reference sentences (strings).
    - candidate_list (list): A list of candidate sentences (strings).
    
    Returns:
    - float: The average BLEU score.
    """
    if len(reference_list) != len(candidate_list):
        raise ValueError("Reference and candidate lists must have the same length.")

    total_bleu_score = 0
    smoothing_function = SmoothingFunction().method1

    for reference, candidate in zip(reference_list, candidate_list):
        reference_tokens = [tokenize_korean(reference)]  # Tokenize reference
        candidate_tokens = tokenize_korean(candidate)   # Tokenize candidate
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)
        total_bleu_score += score

    # Return the average BLEU score across all sentences
    average_bleu_score = round(total_bleu_score / len(reference_list) * 100, 2)
    return average_bleu_score


def calculate_chrf(reference_list, candidate_list):
    """
    Calculate CHRF score for Korean sentences.
    
    Parameters:
    - reference_list (list): A list of reference sentences (strings).
    - candidate_list (list): A list of candidate sentences (strings).
    
    Returns:
    - float: The average CHRF score.
    """
    if len(reference_list) != len(candidate_list):
        raise ValueError("Reference and candidate lists must have the same length.")

    total_chrf_score = 0

    for reference, candidate in zip(reference_list, candidate_list):
        score = sentence_chrf(reference, candidate)
        total_chrf_score += score

    # Return the average CHRF score across all sentences
    average_chrf_score = round(total_chrf_score / len(reference_list) * 100, 2)
    return average_chrf_score

model_name = sys.argv[1]
data_path = Path("data")
llm_a = LLM(model=model_name, dtype="half", max_model_len=MAX_GENERATE_TOKENS, gpu_memory_utilization=0.95)
model_name_for_save = model_name.split("/")[-1]
os.makedirs(model_name_for_save, exist_ok=True)

with open(data_path / "gpt4o_answers.jsonl", "r") as f, open(Path(model_name_for_save) / f"gpt4o_bleu.results.json", "w", encoding="utf-8") as f_out:
    reference_sentences = []
    candidate_sentences = []
    
    results = []

    for line in tqdm(f, total=20):
        item = json.loads(line)
        question = item["question"]
        gpt_answer = item["chosen"]
        outputs = llm_a.generate([question], sampling_cot_params)
        llm_a_output: str = outputs[0].outputs[0].text
        
        reference_sentences.append(gpt_answer)
        candidate_sentences.append(llm_a_output)
        
        out_item = {"question": question, f"{model_name}_answer": llm_a_output, "gpt_answer": gpt_answer}
        results.append(out_item)
        
    bleu_score = calculate_bleu(reference_sentences, candidate_sentences)
    chrf_score = calculate_chrf(reference_sentences, candidate_sentences)

    out_json = {
        "bleu_score": bleu_score,
        "chrf_score": chrf_score,
        "details": results
    }
    
    json.dump(out_json, f_out, ensure_ascii=False, indent=4)
