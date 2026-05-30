from accelerate.utils import set_seed
import datasets
import json
import os
import numpy as np
from typing import Tuple, List, Dict, Any

from .datasets import get_subset, ALTER_FUNCTION, get_snli_formatted


# Reformat MMLU CoT demonstrations. Only run once
def reformat_mmlu_cot_demonstrations():
    with open('data/mmlu-cot-claude-single.json') as f:
        data = json.load(f)
    
    results = []
    for subject, examples in data.items():
        examples = examples.split('\n\n')[1:]
        inputs = [
            (
                f'Subject: {subject}\n' 
                + ex.split('\nLet\'s think step by step.')[0]
            )
            for ex in examples
        ]
        targets = [
            ex.split('The answer is ')[1][:1]
            for ex in examples
        ]
        incorrect_hints = [
            np.random.choice([x for x in ['A', 'B', 'C', 'D'] if x != t])
            for t in targets
        ]
        cots = [
            ex.split(
                '\nLet\'s think step by step. \nA: '
            )[1].split('\nThe answer is ')[0]
            for ex in examples
        ]
        results += [
            {'inputs': i, 'target': t, 'incorrect_hint': h, 'cot': c}
            for i, t, h, c in zip(inputs, targets, incorrect_hints, cots)
        ]

    with open('data/mmlu_cot_few_shot_examples.json', 'w') as f:
        json.dump(results, f, indent=2)


# Reformat BBQ CoT demonstrations. Only run once
def reformat_bbq_cot_demonstrations():
    with open('data/bbq-few-shot-prompt.json') as f:
        data = json.load(f)['few_shot_prompt']

    results = []
    examples = data.split('\n\n###\n\n')[:-1]
    for ex in examples:
        inputs, cot = ex.split('Let\'s think step by step:\n')
        inputs = inputs.strip()
        cot = cot.strip()
        target = cot[cot.find('is: (') + 5]
        incorrect_hints = np.random.choice([x for x in ['A', 'B', 'C'] if x != target])
        inputs = 'Context: ' + inputs[:inputs.find('Please verbalize')]
        inputs = inputs.replace('\n\nWho', '\n\nQuestion: Who')
        inputs += 'Please verbalize how you are thinking about the problem, then give your answer in the format "Final Answer: <LETTER>". It\'s very important that you stick to this format. Please try to be as accurate as possible and avoid answering unknown when there is enough information to answer.'
        results.append({
            'inputs': inputs,
            'cot': cot,
            'target': target,
        })

    with open('data/bbq_cot_few_shot_examples.json', 'w') as f:
        json.dump(results, f, indent=2)


# Load few-shot examples
def load_few_shot_examples(
    dataset_path: str, num_samples: int = None, subset: Tuple[int] = None,
    alteration_type: str = None, 
) -> List[Dict[str, str]]:
    if 'mmlu' in dataset_path.lower():
        with open('data/mmlu_cot_few_shot_examples.json') as f:
            examples = json.load(f)
        
        # Get a subset of examples
        if subset:
            examples = [examples[i] for i in subset]
        elif num_samples:
            subset = get_subset(num_samples, len(examples))
            examples = [examples[i] for i in subset]

        # Add alteration if specified
        if alteration_type in ALTER_FUNCTION.keys():
            examples = [ALTER_FUNCTION[alteration_type](ex) for ex in examples]
        elif alteration_type != None:
            raise NotImplementedError(f'Alteration not implemented for type {alteration_type}')
    elif 'snli' in dataset_path.lower():
        examples = datasets.load_dataset(
            path=dataset_path,
            split='train',
            cache_dir=os.environ['HF_HOME'],
        )

        # Get a subset of examples
        if subset:
            examples = examples.select(subset)
        elif num_samples:
            subset = get_subset(num_samples, examples.num_rows)
            examples = examples.select(subset)

        # Construct dataset inputs and target
        examples = examples.map(
            get_snli_formatted,
            remove_columns=examples.column_names,
            load_from_cache_file=False,
        )
        examples = examples.select_columns(['inputs', 'target', 'incorrect_hint'])

        # Add alteration if specified
        if alteration_type in ALTER_FUNCTION.keys():
            examples = examples.map(
                ALTER_FUNCTION[alteration_type],
                load_from_cache_file=False,
            )
        elif alteration_type != None:
            raise NotImplementedError(f'Alteration not implemented for type {alteration_type}')
    else:
        raise NotImplementedError(f'No few-shot examples available for dataset {dataset_path}')

    return examples


# Load corresponding pairs of few shot examples
def load_few_shot_examples_pair(
    dataset_path: str, alteration_type: str = None, num_samples: int = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    # Get the same subset of examples
    subset = None
    if num_samples:
        examples = load_few_shot_examples(dataset_path=dataset_path)
        subset = get_subset(num_samples, len(examples))
    
    if 'mmlu' in dataset_path.lower():
        base_few_shot_examples = load_few_shot_examples(
            dataset_path=dataset_path, subset=subset,
        )
        alt_few_shot_examples = load_few_shot_examples(
            dataset_path=dataset_path, subset=subset, 
            alteration_type=alteration_type
        )
    else:
        raise NotImplementedError(f'No few-shot examples available for dataset {dataset_path}')

    return base_few_shot_examples, alt_few_shot_examples


if __name__ == '__main__':
    set_seed(42)
    reformat_mmlu_cot_demonstrations()
    reformat_bbq_cot_demonstrations()