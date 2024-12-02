import re
import json
import os
import sys

from tqdm import tqdm
from typing import List, Dict
from pathlib import Path
from litellm import batch_completion


os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
MAX_GENERATE_TOKENS = 2048


def eval_answers_queries(question: str, answer_a: str, answer_b: str):
    lf_judge_sys_prompt = """
    Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. \
    You should choose the assistant that follows the user’s instructions and answers the user’s question better. \
    Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. \
    Begin your evaluation by comparing the two responses and provide a short explanation. \
    Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. \
    Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. \
    After providing your explanation, output your final verdict by strictly following this format: "FINAL ANSWER: [[A]]" if assistant A is better, "FINAL ANSWER: [[B]]" if assistant B is better, and "FINAL ANSWER: [[C]]" for a tie. \ 
    Generate answers in fluent Korean.
    """
    lf_judge_usr_prompt = """
    [User Question]
    {question}
    [The Start of Assistant A’s Answer]
    {answer_a}
    [The End of Assistant A’s Answer]
    [The Start of Assistant B’s Answer]
    {answer_b}
    [The End of Assistant B’s Answer]
    """
    batch_queries = []

    messages = [
            {"content": lf_judge_sys_prompt, "role": "system"},
            {"content": lf_judge_usr_prompt.format(question=question, answer_a = answer_a, answer_b = answer_b), "role": "user"}
    ]
    batch_queries.append(messages)
    return batch_queries

def eval_answers(batch_queries):
    responses = batch_completion(
        model="gpt-4o",
        messages = batch_queries
    )
    resps = [i.choices[0].message.content for i in responses] 

    gpt4_choices = []
    for resp in resps:
        match = re.search(r"\s*\[\[(.*?)\]\]", resp)
    
        # Extract the matched content
        if match:
            final_answer = match.group(1)
            print("Winner: ", final_answer)
            gpt4_choices.append((resp, final_answer))
        else:
            gpt4_choices.append("No FINAL ANSWER")
            print("No FINAL ANSWER found in the string.")
            print("No FINAL ANSWER found in the string.")
    return gpt4_choices

model_resp_dir1 = sys.argv[1]
model_resp_dir2 = "gpt4-o1-preview"
mode = sys.argv[2]
assert mode in ["upper", "lower"]

model_resp_path1 = Path(model_resp_dir1) / "gpt4o_bleu.results.json"
model_resp_path2 = Path(model_resp_dir2) / "gpt4o_bleu.results.json"

with open(model_resp_path1) as f1, open(model_resp_path2) as f2, open(Path(model_resp_dir1) / f"llm_judge_results_{mode}.json", "w") as f_out:
    data1 = json.load(f1)
    data2 = json.load(f2)
    
    items1: List[Dict[str, str]] = data1["details"]
    items2: List[Dict[str, str]] = data2["details"]
    
    num_model_a_win = 0
    num_model_b_win = 0
    num_tie = 0
    num_invalid = 0
    out_items = []
    
    for item1, item2 in tqdm(zip(items1, items2), total=len(items1)):
        if mode == "upper":
            question = item1.pop("question")
            if model_resp_dir1 == "gpt4":
                model_a_name = "gpt4"
                llm_a_output = item1.pop("gpt_answer")
            else:
                item1.pop("gpt_answer")
                model_a_name, llm_a_output = item1.popitem()
                assert len(item1) == 0

            question = item2.pop("question")
            item2.pop("gpt_answer")
            model_b_name, llm_b_output = item2.popitem()
            assert len(item2) == 0
        elif mode == "lower":
            question = item1.pop("question")
            if model_resp_dir1 == "gpt4":
                model_b_name = "gpt4"
                llm_b_output = item1.pop("gpt_answer")
            else:
                item1.pop("gpt_answer")
                model_b_name, llm_b_output = item1.popitem()
                assert len(item1) == 0

            question = item2.pop("question")
            item2.pop("gpt4")
            model_a_name, llm_a_output = item2.popitem()
            assert len(item2) == 0
        else:
            raise ValueError("only supports upper/lower mode")
        
        batch_queries = eval_answers_queries(question=question, answer_a=llm_a_output, answer_b=llm_b_output)
        
        gpt4_choices = eval_answers(batch_queries)
        gpt4_answer, gpt_choice = gpt4_choices[0]
        
        out_item = {
            "gpt4_answer": gpt4_answer,
            "decision": gpt_choice
        }

        out_items.append(out_item)
        
        if "A" in gpt_choice:
            num_model_a_win += 1
        elif "B" in gpt_choice:
            num_model_b_win += 1
        elif "C" in gpt_choice:
            num_tie += 1
        else:
            num_invalid +=1

            
    final_json = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "num_model_a_win": num_model_a_win,
        "num_model_b_win": num_model_b_win,
        "num_tie": num_tie,
        "num_invalid": num_invalid,
        "details": out_items
    }

    json.dump(final_json, f_out, ensure_ascii=False, indent=4)
