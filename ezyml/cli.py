# # # ezyml/cli.py

# # import argparse
# # import pandas as pd
# # from .core import EZTrainer

# # def train_cli(args):
# #     """Handler for the 'train' command."""
# #     print("--- EZYML CLI: Train Mode ---")
# #     try:
# #         trainer = EZTrainer(
# #             data=args.data,
# #             target=args.target,
# #             model=args.model,
# #             task=args.task
# #         )
# #         trainer.train()
        
# #         if args.output:
# #             trainer.save_model(args.output)
        
# #         if args.report:
# #             trainer.save_report(args.report)
            
# #     except Exception as e:
# #         print(f"\nAn error occurred: {e}")

# # def reduce_cli(args):
# #     """Handler for the 'reduce' command."""
# #     print("--- EZYML CLI: Reduce Mode ---")
# #     try:
# #         trainer = EZTrainer(
# #             data=args.data,
# #             model=args.model,
# #             task='dim_reduction',
# #             n_components=args.components
# #         )
# #         trainer.train()
        
# #         if args.output:
# #             trainer.save_transformed(args.output)
            
# #     except Exception as e:
# #         print(f"\nAn error occurred: {e}")


# # def main():
# #     """Main function for the command-line interface."""
# #     parser = argparse.ArgumentParser(description="EZYML: Train and manage ML models easily from the command line.")
# #     subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

# #     # --- Train Command ---
# #     parser_train = subparsers.add_parser("train", help="Train a classification, regression, or clustering model.")
# #     parser_train.add_argument("--data", required=True, help="Path to the input data CSV file.")
# #     parser_train.add_argument("--target", help="Name of the target column (for classification/regression).")
# #     parser_train.add_argument("--model", default="random_forest", help="Name of the model to train.")
# #     parser_train.add_argument("--output", help="Path to save the trained model (.pkl).")
# #     parser_train.add_argument("--report", help="Path to save the evaluation report (.json).")
# #     parser_train.add_argument("--task", default="auto", choices=["auto", "classification", "regression", "clustering"], help="Specify the task type.")
# #     parser_train.set_defaults(func=train_cli)

# #     # --- Reduce Command ---
# #     parser_reduce = subparsers.add_parser("reduce", help="Perform dimensionality reduction.")
# #     parser_reduce.add_argument("--data", required=True, help="Path to the input data CSV file.")
# #     parser_reduce.add_argument("--model", required=True, choices=["pca", "tsne"], help="Dimensionality reduction method.")
# #     parser_reduce.add_argument("--components", type=int, required=True, help="Number of components to reduce to.")
# #     parser_reduce.add_argument("--output", required=True, help="Path to save the transformed data (.csv).")
# #     parser_reduce.set_defaults(func=reduce_cli)

# #     args = parser.parse_args()
# #     args.func(args)

# # if __name__ == '__main__':
# #     main()


# # ezyml/cli.py

# import argparse
# import sys
# import pandas as pd

# from ezyml.core import EZTrainer
# from ezyml.pipeline.loader import load_pipeline
# from ezyml.compiler.compile import compile_project
# from ezyml.eda.auto_eda import auto_eda
# from ezyml.evaluation.evaluator import Evaluator
# from ezyml.monitoring.fingerprint import dataset_fingerprint


# # ------------------------------------------------------------------
# # TRAIN COMMAND
# # ------------------------------------------------------------------

# def train_cli(args):
#     print("\n--- EZYML :: TRAIN ---")

#     trainer = EZTrainer(
#         data=args.data,
#         target=args.target,
#         model=args.model,
#         task=args.task
#     )

#     trainer.train()

#     if args.learn:
#         print("\n[LEARN MODE]")
#         print(f"Model chosen: {args.model}")
#         print("Reason: Good default for tabular data.")

#     if args.output:
#         trainer.save_model(args.output)
#         print(f"Model saved to {args.output}")

#     if args.report:
#         trainer.save_report(args.report)
#         print(f"Report saved to {args.report}")


# # ------------------------------------------------------------------
# # REDUCE COMMAND
# # ------------------------------------------------------------------

# def reduce_cli(args):
#     print("\n--- EZYML :: REDUCE ---")

#     trainer = EZTrainer(
#         data=args.data,
#         model=args.model,
#         task="dim_reduction",
#         n_components=args.components
#     )

#     trainer.train()
#     trainer.save_transformed(args.output)
#     print(f"Reduced data saved to {args.output}")


# # ------------------------------------------------------------------
# # COMPILE COMMAND  (🔥 NEW 🔥)
# # ------------------------------------------------------------------

# def compile_cli(args):
#     print("\n--- EZYML :: COMPILE ---")

#     # 1. Load pipeline
#     pipeline = load_pipeline(args.pipeline)

#     # 2. Load data
#     df = pd.read_csv(args.data)

#     # 3. Auto EDA (optional but default)
#     if not args.no_eda:
#         eda_report = auto_eda(df, target=args.target)
#         print("[EDA] Completed")

#     # 4. Run pipeline
#     print("[PIPELINE] Executing DAG...")
#     # outputs = pipeline.run(df, target=args.target)

#     # # Convention: last node is model trainer
#     # trainer = list(outputs.values())[-1]
#     trainer = pipeline.run(df, target=args.target)


#     # 5. Evaluation
#     evaluator = Evaluator(task="classification")
#     metrics = evaluator.evaluate(
#         trainer.y_test,
#         trainer.y_pred,
#         getattr(trainer, "y_prob", None)
#     )

#     evaluator.save(metrics, out_dir="build")
#     evaluator.visualize(
#         trainer.y_test,
#         trainer.y_pred,
#         getattr(trainer, "y_prob", None),
#         out_dir="build/plots"
#     )

#     # 6. Dataset fingerprint
#     fp = dataset_fingerprint(df)
#     print(f"[FINGERPRINT] {fp}")

#     # 7. Compile artifacts
#     compile_project(
#         model=trainer.model,
#         pipeline=pipeline,
#         schema={c: "number" for c in df.drop(columns=[args.target]).columns},
#         image_name=args.image,
#         replicas=args.replicas,
#         with_demo=not args.no_demo,
#         with_k8s=not args.no_k8s,
#         build_dir="build"
#     )

#     print("\n[SUCCESS] Compilation complete.")
#     print("Artifacts generated in ./build/")


# # ------------------------------------------------------------------
# # MAIN CLI ENTRY
# # ------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(
#         description="EZYML — From dataset to production ML system"
#     )

#     sub = parser.add_subparsers(dest="command", required=True)

#     # ---------------- TRAIN ----------------
#     train = sub.add_parser("train", help="Train a model")
#     train.add_argument("--data", required=True)
#     train.add_argument("--target", required=True)
#     train.add_argument("--model", default="random_forest")
#     train.add_argument("--task", default="classification")
#     train.add_argument("--output")
#     train.add_argument("--report")
#     train.add_argument("--learn", action="store_true")
#     train.set_defaults(func=train_cli)

#     # ---------------- REDUCE ----------------
#     reduce = sub.add_parser("reduce", help="Dimensionality reduction")
#     reduce.add_argument("--data", required=True)
#     reduce.add_argument("--model", choices=["pca", "tsne"], required=True)
#     reduce.add_argument("--components", type=int, required=True)
#     reduce.add_argument("--output", required=True)
#     reduce.set_defaults(func=reduce_cli)

#     # ---------------- COMPILE ----------------
#     compile_cmd = sub.add_parser("compile", help="Compile full ML system")
#     compile_cmd.add_argument("--pipeline", required=True, help="Pipeline YAML")
#     compile_cmd.add_argument("--data", required=True, help="CSV dataset")
#     compile_cmd.add_argument("--target", required=True)
#     compile_cmd.add_argument("--image", default="ezyml-model")
#     compile_cmd.add_argument("--replicas", type=int, default=1)
#     compile_cmd.add_argument("--no-demo", action="store_true")
#     compile_cmd.add_argument("--no-k8s", action="store_true")
#     compile_cmd.add_argument("--no-eda", action="store_true")
#     compile_cmd.set_defaults(func=compile_cli)

#     args = parser.parse_args()
#     args.func(args)


# if __name__ == "__main__":
#     main()

# ezyml/cli.py

import argparse
import pandas as pd

from ezyml.core import EZTrainer
from ezyml.pipeline.loader import load_pipeline
from ezyml.compiler.compile import compile_project
from ezyml.eda.auto_eda import auto_eda
from ezyml.monitoring.fingerprint import dataset_fingerprint


def compile_cli(args):
    print("\n--- EZYML :: COMPILE ---")

    df = pd.read_csv(args.data)

    if not args.no_eda:
        auto_eda(df, target=args.target)
        print("[EDA] Completed")

    pipeline = load_pipeline(args.pipeline)
    trainer = pipeline.run(df, target=args.target)

    fingerprint = dataset_fingerprint(df)
    print(f"[FINGERPRINT] {fingerprint}")

    schema = {c: "number" for c in df.drop(columns=[args.target]).columns}

    # --all overrides everything
    api = args.api or args.all
    demo = args.demo or args.all
    docker = args.docker or args.all
    k8s = args.k8s or args.all

    compile_project(
        trainer=trainer,
        schema=schema,
        api=api,
        demo=demo,
        docker=docker,
        k8s=k8s
    )

    print("\n[SUCCESS] Compilation complete.")
    print("Generated:")
    print(f"  model.pkl")
    print(f"  metrics.json")
    if api: print("  app.py + openapi.json")
    if demo: print("  demo_app.py")
    if docker: print("  Dockerfile")
    if k8s: print("  k8s.yaml")


def main():
    parser = argparse.ArgumentParser("ezyml")
    sub = parser.add_subparsers(dest="command", required=True)

    compile_cmd = sub.add_parser("compile")
    compile_cmd.add_argument("--pipeline", required=True)
    compile_cmd.add_argument("--data", required=True)
    compile_cmd.add_argument("--target", required=True)

    compile_cmd.add_argument("--api", action="store_true")
    compile_cmd.add_argument("--demo", action="store_true")
    compile_cmd.add_argument("--docker", action="store_true")
    compile_cmd.add_argument("--k8s", action="store_true")
    compile_cmd.add_argument("--all", action="store_true")

    compile_cmd.add_argument("--no-eda", action="store_true")
    compile_cmd.set_defaults(func=compile_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
