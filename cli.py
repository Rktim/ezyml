# ezyml/cli.py

import argparse
import pandas as pd
from .core import EZTrainer

def train_cli(args):
    """Handler for the 'train' command."""
    print("--- EZYML CLI: Train Mode ---")
    try:
        trainer = EZTrainer(
            data=args.data,
            target=args.target,
            model=args.model,
            task=args.task
        )
        trainer.train()
        
        if args.output:
            trainer.save_model(args.output)
        
        if args.report:
            trainer.save_report(args.report)
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")

def reduce_cli(args):
    """Handler for the 'reduce' command."""
    print("--- EZYML CLI: Reduce Mode ---")
    try:
        trainer = EZTrainer(
            data=args.data,
            model=args.model,
            task='dim_reduction',
            n_components=args.components
        )
        trainer.train()
        
        if args.output:
            trainer.save_transformed(args.output)
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")


def main():
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser(description="EZYML: Train and manage ML models easily from the command line.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Train Command ---
    parser_train = subparsers.add_parser("train", help="Train a classification, regression, or clustering model.")
    parser_train.add_argument("--data", required=True, help="Path to the input data CSV file.")
    parser_train.add_argument("--target", help="Name of the target column (for classification/regression).")
    parser_train.add_argument("--model", default="random_forest", help="Name of the model to train.")
    parser_train.add_argument("--output", help="Path to save the trained model (.pkl).")
    parser_train.add_argument("--report", help="Path to save the evaluation report (.json).")
    parser_train.add_argument("--task", default="auto", choices=["auto", "classification", "regression", "clustering"], help="Specify the task type.")
    parser_train.set_defaults(func=train_cli)

    # --- Reduce Command ---
    parser_reduce = subparsers.add_parser("reduce", help="Perform dimensionality reduction.")
    parser_reduce.add_argument("--data", required=True, help="Path to the input data CSV file.")
    parser_reduce.add_argument("--model", required=True, choices=["pca", "tsne"], help="Dimensionality reduction method.")
    parser_reduce.add_argument("--components", type=int, required=True, help="Number of components to reduce to.")
    parser_reduce.add_argument("--output", required=True, help="Path to save the transformed data (.csv).")
    parser_reduce.set_defaults(func=reduce_cli)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
