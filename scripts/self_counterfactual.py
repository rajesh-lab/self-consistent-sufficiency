from accelerate.utils import set_seed
import argparse
from datasets import Dataset
import json
import numpy as np

from src.datasets import load_dataset
from src.fewshots import load_few_shot_examples
from src.models import load_llm, generate_outputs
from src.outputs import save_results, batch_extract_answers, compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        prog='SelfCounterfactual',
        description='LLM self-counterfactual explanations test',
    )
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--alteration-type', type=str)
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--num-few-shot-examples', type=int)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    batch_size = 8

    # Set random seed
    set_seed(args.seed)

    # Load model
    print('Loading model ...')
    model, tokenizer = load_llm(model_name=args.model_name)
    generation_kwargs = {
        'max_new_tokens': 2048,
        'do_sample': False,
        'temperature': None,
        'top_p': None,
        'top_k': None,
        'repetition_penalty': 1.2,
    }
    alternative_kwargs = {
        'max_new_tokens': 2048,
        'do_sample': True,
        'temperature': 0.1,
        'top_p': 0.95,
        'top_k': 50,
        'repetition_penalty': 1.2,
    }

    # Load datasets
    print('Loading datasets ...')
    base_dataset, task_prompt, subset = load_dataset(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        alteration_type=args.alteration_type,
        num_samples=args.num_samples,
    )

    base_few_shot_examples = None
    if args.num_few_shot_examples:
        base_few_shot_examples = load_few_shot_examples(
            dataset_path=args.dataset_path,
            alteration_type=args.alteration_type,
            num_samples=args.num_few_shot_examples,
        )

    # Generate base outputs
    print()
    print('Generating base outputs ...')
    base_generations, _ = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        dataset=base_dataset,
        task_prompt=task_prompt,
        generation_kwargs=generation_kwargs,
        use_cot=True,
        few_shot_examples=base_few_shot_examples,
        batch_size=batch_size,
        progress_bar=True,
    )
    base_answers, _ = batch_extract_answers(
        base_generations, use_cot=True
    )
    base_targets = [ex['target'] for ex in base_dataset]

    # Generate self-counterfactual
    print()
    print('Generating self-counterfactuals ...')
    self_counterfactual_prompt = '''You are an expert-level evaluator. You will be given a passage with a question.
Your task is to edit the passage such that the answer to the passage is different from how it should currently be answered.
Make as few edits as possible. First briefly reason about how you would edit the passage. Then produce your altered passage.

At the end, clearly state your edited passage between the tags Start Of Passage and End Of Passage.'''
    self_counterfactual_dataset = Dataset.from_dict({
        'inputs': [
            f'''Start Of Passage:\n{ex["inputs"]}\nEnd Of Passage
Edit the passage above such that the answer to the passage is different from how it should currently be answered.
Make as few edits as possible. First briefly reason about how you would edit the passage. Then produce your altered passage.'''
            for ex in base_dataset
        ],
    })
    self_counterfactual_generations, _ = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        dataset=self_counterfactual_dataset,
        task_prompt=self_counterfactual_prompt,
        generation_kwargs=alternative_kwargs,
        use_cot=True,
        batch_size=batch_size,
        progress_bar=True,
    )
    self_counterfactuals, _ = batch_extract_answers(
        self_counterfactual_generations, use_cot=True, 
        prefix='Start Of Passage', suffix='End Of Passage', normalize=False,
    )

    # Generate alternative answers
    print()
    print('Generating alternative answers ...')
    self_counterfactual_dataset = Dataset.from_dict({
        'inputs': [f'{sc}' for sc in self_counterfactuals],
    })
    alt_generations, _ = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        dataset=self_counterfactual_dataset,
        task_prompt=task_prompt,
        generation_kwargs=generation_kwargs,
        use_cot=True,
        few_shot_examples=base_few_shot_examples,
        batch_size=batch_size,
        progress_bar=True,
    )
    alt_answers, _ = batch_extract_answers(alt_generations, use_cot=True)

    # Compute metrics
    base_answers = np.array(base_answers)
    alt_answers = np.array(alt_answers)
    base_targets = np.array(base_targets)

    base_accuracy_array = compute_accuracy(
        preds=base_answers, targets=base_targets, return_average=False,
    )
    base_accuracy = base_accuracy_array.mean()
    answer_change_array = (base_answers != alt_answers)
    answer_change = answer_change_array.mean()
    answer_change_gv_correct_base = answer_change_array[
        base_accuracy_array
    ].mean()

    # Save results
    name = f'dataset_{args.dataset_path}-alteration_{args.alteration_type}-model_{args.model_name}-num_samples_{args.num_samples}-self_counterfactuals'
    results = {
        'base_accuracy': float(base_accuracy),
        'answer_change': float(answer_change),
        'answer_change_gv_correct_base': float(answer_change_gv_correct_base),
    }
    save_results(results=results, name=name, file_name='data/results.json')
    print(json.dumps(results))