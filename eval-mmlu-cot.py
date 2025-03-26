import numpy as np
import pandas as pd
import argparse
import re

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from datasets import load_dataset

# For JSON decoding
from pydantic import BaseModel
from typing import Annotated, Union, Literal

# Define answer choices and reasoning generation
ValidAnswers = Annotated[Union[
    Literal["A"], 
    Literal["B"], 
    Literal["C"], 
    Literal["D"]
], "Valid answers"]

class ResoningGeneration(BaseModel):
    resoning_steps: str
    final_answer: ValidAnswers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--model', type=str, help='Model to evaluate', default="/cfs/collections/llms/models/instruct/Llama-3.2-1B-Instruct")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--util', type=float, default=0.85)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"output/eval_mmlu_cot_{args.model.split('/')[-1]}.csv"
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
         'content': "You are an expert across many domains. You will be posed a multiple-choice question, from which you will select the correct answer (only one possible answer). You will first think on the answer step-by-step, before arriving at a conclusion. You will then have a final answer with 'A', 'B', 'C', or 'D'. You will be given a score based on the number of correct answers. Do your absolute best. Your output should be in the following JSON format: {'resoning_steps': your resoning steps to arrive at the answer, 'final_answer': 'A', 'B', 'C', or 'D'}."},
        {'role': 'user',
         'content': prompt},
    ], tokenize=False)

    example['prompt'] = prompt
    return example

def main():
    args = parse_args()

    # Load model
    llm = LLM(model=args.model, 
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True,
              gpu_memory_utilization=args.util,)

    tokenizer = llm.get_tokenizer()

    guided_decoding_params = GuidedDecodingParams(json=ResoningGeneration.model_json_schema())
    sampling_params = SamplingParams(temperature=0, max_tokens=1000,
    guided_decoding=guided_decoding_params)

    # Load dataset and prepare prompt
    ds = load_dataset("cais/mmlu", "astronomy", split='test')
    ds = ds.map(create_prompt, fn_kwargs={'tokenizer': tokenizer})

    # Generate output
    outputs = llm.generate(ds['prompt'], sampling_params)
    output_text = [o.outputs[0].text.strip() for o in outputs]

    # Save to CSV
    ds = ds.to_pandas()
    ds['output'] = output_text
    ds.to_csv(args.output, index=False)
    
if __name__ == '__main__':
    main()