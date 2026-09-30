"""
Augment 'Post-training-Data-Flywheel/gpt4-self-instruct' with multi-constraints for RLVR.
Implements the methodology from 'Generalizing Verifiable Instruction Following' (AI2 - IFBench / IF-RLVR).
"""

import sys
import os
import argparse

# Add parent directory to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from augment_with_constraints import run_augmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment Post-training-Data-Flywheel/gpt4-self-instruct with verifiable constraints.")
    parser.add_argument("num_samples", type=int, nargs="?", default=1000, help="Target total samples or count to generate")
    parser.add_argument("--resume", action="store_true", help="Resume from existing file and continue until target count")
    parser.add_argument("--append", action="store_true", help="Append new samples to existing file")
    parser.add_argument("--split", type=str, default="seen", choices=["seen", "unseen"], help="Constraint split: seen or unseen")
    parser.add_argument("--output", type=str, default="datasets/augmented_gpt4_self_instruct.jsonl", help="Output path")
    args = parser.parse_args()

    dataset_name = "Post-training-Data-Flywheel/gpt4-self-instruct"
    output_path = args.output
    
    run_augmentation(
        dataset_name=dataset_name,
        output_path=output_path,
        num_samples=args.num_samples,
        resume=args.resume,
        append=args.append,
        constraint_split=args.split
    )