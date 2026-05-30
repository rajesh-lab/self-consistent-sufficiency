import datasets
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Dict, List, Any


# Load tokenizer and causal LM on multiple GPUs
def load_llm(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map='auto'
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    return model, tokenizer


# Generate model's outputs
def generate_outputs(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    dataset: datasets.arrow_dataset.Dataset, 
    task_prompt: str,
    generation_kwargs: Dict[str, Any],
    use_cot: bool = True,
    few_shot_examples: datasets.arrow_dataset.Dataset = None,
    batch_size: int = None,
    progress_bar: bool = False,
) -> Tuple[List[str], List[str]]:
    device = model.device
    generations = []
    prompts = []

    # Detect if using Qwen
    is_qwen = 'qwen' in tokenizer.name_or_path.lower()
        
    # Construct system prompt
    system_prompt = ''
    if is_qwen:
        system_prompt = 'You are Qwen, created by Alibaba Cloud. '
    system_prompt += f'You are a helpful assistant. {task_prompt}'

    # Construct few-shot examples
    few_shot_messages = []
    if few_shot_examples:
        for ex in few_shot_examples:
            if 'cot' in ex.keys():
                output = (
                    f'Let\'s think step by step: {ex["cot"]}\n'
                    + f'Final Answer: {ex["target"]}'
                )
            else:
                output = f'Final Answer: {ex["target"]}'
            few_shot_messages += [
                {'role': 'user', 'content': ex['inputs']},
                {'role': 'assistant', 'content': output}
            ]

    # Iterate in batches
    if batch_size is None:
        batch_size = dataset.num_rows
    if progress_bar:
        iteration = tqdm(range(0, dataset.num_rows, batch_size), desc='Generating outputs')
    else:
        iteration = range(0, dataset.num_rows, batch_size)
    for start_idx in iteration:
        batch = dataset.select(list(range(
            start_idx, min(start_idx + batch_size, dataset.num_rows)
        )))

        # Construct batch messages
        texts = []
        for ex in batch:
            # Construct user prompt
            user_prompt = ex['inputs']
            if is_qwen:
                user_prompt += ' /no_think' # if not use_cot else ' /think'

            # Construct chat message
            messages = [
                {'role': 'system', 'content': system_prompt},
                *few_shot_messages,
                {'role': 'user', 'content': user_prompt},
            ]

            # Apply tokenizer chat template
            special_kwargs = {}
            if is_qwen:
                special_kwargs = {'enable_thinking': False}
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **special_kwargs
            )

            # If use CoT, add step-by-step prefix
            if use_cot:
                text += 'Let\'s think step by step: '

            texts.append(text)
        
        # Tokenize the batch
        model_inputs = tokenizer(
            texts, return_tensors='pt', padding=True
        ).to(device)

        # Generate batch outputs
        generated_ids = model.generate(**model_inputs, **generation_kwargs)
        if 'num_return_sequences' in generation_kwargs.keys():
            num_return_sequences = generation_kwargs['num_return_sequences']
            assert generated_ids.shape[0] == num_return_sequences * len(batch)

        # Decode each output
        for i in range(generated_ids.shape[0]):
            if 'num_return_sequences' in generation_kwargs.keys():
                prompt_len = model_inputs.input_ids[
                    i//num_return_sequences
                ].shape[0]
            else:
                prompt_len = model_inputs.input_ids[i].shape[0]
            prompt_ids = generated_ids[i][:prompt_len]
            output_ids = generated_ids[i][prompt_len:]
            prompt = tokenizer.decode(
                prompt_ids, skip_special_tokens=True
            ).strip('\n')
            output = tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip('\n')
            prompts.append(prompt)
            generations.append(output)

    return generations, prompts