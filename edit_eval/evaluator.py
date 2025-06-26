import os
import json
import math
import torch
import numpy as np

from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple

from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig


GENERATION_TYPES = [
    "REWRITE",
    "REPHRASE",
    "LOCALITY",
    "SBJ",
    "OBJ_NEW",
    "OBJ_OLD",
    "SBJ_HOP",
    "OBJ_NEW_HOP",
    "OBJ_OLD_HOP",
]


class TextEvaluator:
    @staticmethod
    def calculate_repetition_scores(text: str) -> Dict[str, float]:
        def get_tokens(text: str) -> List[str]:
            return text.replace('\n', ' ').strip().split()

        def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
            return [(token,) for token in tokens] if n == 1 else list(zip(*[tokens[i:] for i in range(n)]))

        def calc_segment_repetition(tokens: List[str], n: int) -> float:
            ngrams = get_ngrams(tokens, n)
            if not ngrams:
                return 0
            counter = Counter(ngrams)
            return sum(count - 1 for count in counter.values() if count > 1)

        tokens = get_tokens(text)
        if not tokens:
            return {"SRS": 0}

        srs_scores = [calc_segment_repetition(tokens, n) for n in range(1, 5)]
        return {"SRS": sum(srs_scores)}


def prepare_prompts(case: Dict[str, Any], base_prompt: str) -> Dict[str, List[str]]:
    prompts: Dict[str, List[str]] = {
        "REWRITE": [base_prompt],
        "REPHRASE": [case.get("rephrased_prompt", "")],
        "LOCALITY": [case["locality_prompt"]] if case.get("locality_prompt") else []
    }

    key_map = {
        "sbj_hop_sentence":    "SBJ_HOP",
        "obj_old_hop_sentence": "OBJ_OLD_HOP",
        "obj_new_hop_sentence": "OBJ_NEW_HOP",
        "sbj_sentence":         "SBJ",
        "obj_old_sentence":     "OBJ_OLD",
        "obj_new_sentence":     "OBJ_NEW"
    }

    for json_key, group_name in key_map.items():
        sentences = case.get(json_key, [])
        if sentences:
            prompts[group_name] = [f"{s} {base_prompt}" for s in sentences]

    return {k: v for k, v in prompts.items() if v and v[0].strip()}


def process_evaluation_result(
    result: Dict[str, Any],
    gen_type: str,
    results_by_type: Dict[str, List],
    metrics_by_type: Dict[str, Dict],
    case_analysis: Dict[str, Dict]
) -> None:
    results_by_type[gen_type].append(result)
    metrics_by_type[gen_type]['editing_success'].append(result.get('editing_success', False))

    if gen_type == "REWRITE" and "repetition_scores" in result:
        metrics_by_type[gen_type]['repetition_SRS'].append(result['repetition_scores'].get('SRS', 0))

    if gen_type == "LOCALITY":
        gt = result.get('locality_ground_truth', '').lower().strip()
        out = result.get('generated_text', '').lower().strip()
        success = gt in out
        result['editing_success'] = success
        result['locality_success'] = success
        metrics_by_type[gen_type]['locality_success'].append(success)
        result.setdefault('analysis', {})['locality_check'] = {
            'ground_truth': gt,
            'generated_text': out,
            'success': success
        }

    case_result = {
        'case_id': result.get('case_id', 'unknown'),
        'generation_type': gen_type,
        'generated_text': result.get('generated_text', ''),
        'edited_knowledge': result['edited_knowledge'],
        'fact_knowledge': result['fact_knowledge'],
        'prompt': result['prompt'],
        'editing_success': result.get('editing_success', False),
        'analysis': {}
    }
    if 'repetition_scores' in result:
        case_result['analysis']['repetition_scores'] = result['repetition_scores']

    key = 'success_cases' if result.get('editing_success', False) else 'failure_cases'
    case_analysis[gen_type][key].append(case_result)


def calculate_average_metrics(metrics_by_type: Dict[str, Dict]) -> Dict[str, Dict]:
    avg = {}
    for gen_type, metrics in metrics_by_type.items():
        if not metrics:
            continue
        avg[gen_type] = {}
        for name, vals in metrics.items():
            if not vals:
                continue
            if isinstance(vals[0], (bool, np.bool_)):
                avg[gen_type][name] = sum(vals) / len(vals)
            else:
                avg[gen_type][name] = sum(vals) / len(vals)
    return avg


def evaluate_edited_model(
    model_path: str,
    model_name: str,
    test_data: List[Dict],
    max_length: int = 60,
    batch_size: int = 512,
    cuda_device: str = "0",
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    evaluator = TextEvaluator()
    results_by_type = defaultdict(list)
    metrics_by_type = defaultdict(lambda: defaultdict(list))
    case_analysis = defaultdict(lambda: {'success_cases': [], 'failure_cases': []})

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_conf = GenerationConfig.from_pretrained(model_path)

    prompts, p2t, p2c = [], {}, {}
    for case in test_data:
        bp = case['prompt'].format(case['subject']) if '{}' in case['prompt'] else case['prompt']
        for t, lst in prepare_prompts(case, bp).items():
            for pr in lst:
                p2t[pr] = t
                p2c[pr] = case
                prompts.append(pr)

    max_prompt_len = max(len(tokenizer.encode(pr)) for pr in prompts)
    total_seq_len = max_prompt_len + max_length + 8

    if prompts:
        vllm = LLM(
            model=model_path,
            gpu_memory_utilization=0.75,
            max_num_seqs=batch_size,
            max_model_len=total_seq_len,
            max_num_batched_tokens=total_seq_len * batch_size,
            swap_space=8,
            dtype='float16'
        )
        samp = SamplingParams(
            max_tokens=max_length,
            temperature=gen_conf.temperature,
            top_p=gen_conf.top_p,
            top_k=gen_conf.top_k,
            stop=None,
            stop_token_ids=[],
            ignore_eos=True,
        )
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing prompts"):
            batch = prompts[i:i+batch_size]
            try:
                outs = vllm.generate(batch, samp)
                for pr, out in zip(batch, outs):
                    t = p2t[pr]
                    c = p2c[pr]
                    txt = out.outputs[0].text
                    res = {
                        'case_id': c['case_id'],
                        'prompt': pr,
                        'generated_text': txt,
                        'edited_knowledge': c['edited_knowledge'],
                        'fact_knowledge': c['fact_knowledge']
                    }
                    if t == 'LOCALITY':
                        res['locality_ground_truth'] = c.get('locality_ground_truth', '')
                    lower = txt.lower()
                    success = (c['edited_knowledge'].lower() in lower
                               and c['fact_knowledge'].lower() not in lower)
                    res['editing_success'] = success
                    if t == 'REWRITE':
                        res['repetition_scores'] = evaluator.calculate_repetition_scores(txt)
                    process_evaluation_result(res, t, results_by_type, metrics_by_type, case_analysis)
            except RuntimeError as e:
                print(f"Error processing batch: {e}")
        del vllm
        torch.cuda.empty_cache()

    avg_metrics = calculate_average_metrics(metrics_by_type)
    return results_by_type, avg_metrics
