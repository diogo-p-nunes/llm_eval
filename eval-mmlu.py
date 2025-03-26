import numpy as np
import pandas as pd
import argparse

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--model', type=str, help='Model to evaluate', default="/cfs/collections/llms/models/instruct/Llama-3.2-1B-Instruct")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--util', type=float, default=0.85)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"output/eval_mmlu_{args.model.split('/')[-1]}.csv"
    return args

def create_prompt(example, tokenizer):
    # Create prompt from question and choices
    prompt = f"Question: {example['question']}\nChoices:\n"
    for label, choice in zip(['A', 'B', 'C', 'D'], example['choices']):
        prompt += f"{label}. {choice}\n"
    prompt += f"Answer: "

    # Apply chat template
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 
         'content': "You are an expert across many domains. You will be posed a multiple-choice question, from which you will select the correct answer (only one possible answer). You only answer with 'A', 'B', 'C', or 'D'. You will be given a score based on the number of correct answers. Do your absolute best."},
        {'role': 'user',
         'content': prompt},
    ], tokenize=False)

    example['prompt'] = prompt
    return example

def convert_options(options):
    mapper = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return [mapper[o] for o in options]

def main():
    args = parse_args()

    # Load model
    llm = LLM(model=args.model, 
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True,
              gpu_memory_utilization=args.util,)

    tokenizer = llm.get_tokenizer()

    guided_decoding_params = GuidedDecodingParams(choice=["A", "B", "C", "D"])
    sampling_params = SamplingParams(temperature=0, guided_decoding=guided_decoding_params)

    # Load dataset and prepare prompt
    ds = load_dataset("cais/mmlu", "all", split='test')
    ds = ds.map(create_prompt, fn_kwargs={'tokenizer': tokenizer})

    # Generate output
    outputs = llm.generate(ds['prompt'], sampling_params)
    output_text = [o.outputs[0].text.strip() for o in outputs]

    # Save to CSV
    ds = ds.to_pandas()
    ds['output'] = output_text
    ds['correct'] = np.equal(convert_options(ds['output']), ds['answer'])
    ds.to_csv(args.output, index=False)

    # Print scores
    print(f"Overall: {ds['correct'].mean()}")
    print(ds.groupby('subject')['correct'].mean())
    
if __name__ == '__main__':
    main()