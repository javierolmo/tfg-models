"""Unified Command Line Interface for training, evaluating, and predicting property models."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import settings
from core.model_handler import AzureModelHandler, LocalModelHandler
from data.providers import AzureDataProvider, LocalDataProvider
from models import MODEL_REGISTRY, get_model_trainer

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tfg-models")


def get_model_handler(handler_arg: str, model_name: str):
    """Instantiates ModelHandler without initializing Spark."""
    if handler_arg == "azure":
        return AzureModelHandler(model_name)
    return LocalModelHandler(model_name)


def get_data_provider(provider_arg: str):
    """Instantiates DataProvider (initializes Spark when reading)."""
    if provider_arg == "azure":
        return AzureDataProvider()
    return LocalDataProvider()


def handle_train(args):
    """Handles the 'train' subcommand."""
    models_to_train = (
        ["linear_regression", "lightgbm"] if args.model == "all" else [args.model]
    )

    results = []
    # Initialize provider once for all models to share the same session/data
    data_provider = get_data_provider(args.provider)

    for model_name in models_to_train:
        logger.info("Initializing trainer for: %s", model_name)
        handler = get_model_handler(args.handler, model_name)
        trainer = get_model_trainer(
            model_name,
            data_provider=data_provider,
            model_handler=handler,
        )
        report = trainer.run(save=True)
        results.append(report)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY REPORT")
    print("=" * 60)
    for r in results:
        print(f"Model: {r.get('model_name')}")
        print(f"  Version:  {r.get('saved_version')}")
        print(f"  MAE:      {r.get('mae', 0):,.2f} €")
        print(f"  RMSE:     {r.get('rmse', 0):,.2f} €")
        if "r2_score" in r:
            print(f"  R2 Score: {r.get('r2_score'):.4f}")
        print("-" * 60)


def handle_predict(args):
    """Handles the 'predict' subcommand (fast inference without Spark)."""
    handler = get_model_handler(args.handler, args.model)
    trainer = get_model_trainer(args.model, model_handler=handler)

    price = trainer.predict_sample(
        surface=args.surface,
        rooms=args.rooms,
        bathrooms=args.bathrooms,
        postal_code=args.postal_code,
        elevator=args.elevator,
        terrace=args.terrace,
        garage=args.garage,
        version=args.version,
    )

    print("\n" + "=" * 50)
    print(f"PROPERTY VALUATION RESULT ({args.model.upper()})")
    print("=" * 50)
    print(f"  Postal Code: {args.postal_code}")
    print(f"  Surface:     {args.surface} m²")
    print(f"  Rooms:       {args.rooms} | Bathrooms: {args.bathrooms}")
    print(f"  Features:    Elevator={args.elevator}, Terrace={args.terrace}, Garage={args.garage}")
    print("-" * 50)
    print(f"  ESTIMATED PRICE: {price:,.2f} €")
    print("=" * 50 + "\n")


def handle_compare(args):
    """Compares the latest evaluation reports of trained models (instantaneous)."""
    models = ["linear_regression", "lightgbm"]
    print("\n" + "=" * 70)
    print(f"{'MODEL':<20} | {'MAE (€)':<12} | {'RMSE (€)':<12} | {'R²':<8} | {'VERSION':<15}")
    print("-" * 70)

    for m in models:
        handler = get_model_handler(args.handler, m)
        try:
            report = handler.get_report(version="latest")
            mae = f"{report.get('mae', 0):,.2f}"
            rmse = f"{report.get('rmse', 0):,.2f}"
            r2 = f"{report.get('r2_score', 0):.4f}" if "r2_score" in report else "N/A"
            version = str(report.get("version", "latest"))[:14]
            print(f"{m:<20} | {mae:<12} | {rmse:<12} | {r2:<8} | {version:<15}")
        except Exception as e:
            print(f"{m:<20} | (No trained version found: {e})")
    print("=" * 70 + "\n")


def handle_versions(args):
    """Lists saved model versions."""
    handler = get_model_handler(args.handler, args.model)
    versions = handler.list_versions()
    print(f"\nAvailable versions for '{args.model}':")
    if not versions:
        print("  (None found)")
    for v in versions:
        print(f"  - {v}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="TFG Real Estate Valuation Models CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train property valuation models")
    train_parser.add_argument(
        "--model",
        choices=["linear_regression", "lightgbm", "all"],
        default="all",
        help="Model architecture to train",
    )
    train_parser.add_argument(
        "--provider",
        choices=["azure", "local"],
        default=settings.default_data_provider,
        help="Data provider to read properties from",
    )
    train_parser.add_argument(
        "--handler",
        choices=["azure", "local"],
        default=settings.default_model_handler,
        help="Target storage for persisting the trained model",
    )
    train_parser.set_defaults(func=handle_train)

    # predict
    predict_parser = subparsers.add_parser("predict", help="Predict property valuation")
    predict_parser.add_argument(
        "--model",
        choices=["linear_regression", "lightgbm"],
        default="lightgbm",
        help="Model to use for prediction",
    )
    predict_parser.add_argument("--surface", type=int, required=True, help="Surface in m²")
    predict_parser.add_argument("--rooms", type=int, required=True, help="Number of rooms")
    predict_parser.add_argument("--bathrooms", type=int, required=True, help="Number of bathrooms")
    predict_parser.add_argument("--postal-code", type=int, required=True, help="Postal code")
    predict_parser.add_argument("--elevator", action="store_true", help="Has elevator")
    predict_parser.add_argument("--terrace", action="store_true", help="Has terrace")
    predict_parser.add_argument("--garage", action="store_true", help="Has garage")
    predict_parser.add_argument("--version", default="latest", help="Model version to use")
    predict_parser.add_argument(
        "--handler",
        choices=["azure", "local"],
        default=settings.default_model_handler,
    )
    predict_parser.set_defaults(func=handle_predict)

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare performance of trained models")
    compare_parser.add_argument(
        "--handler",
        choices=["azure", "local"],
        default=settings.default_model_handler,
    )
    compare_parser.set_defaults(func=handle_compare)

    # versions
    versions_parser = subparsers.add_parser("versions", help="List saved versions for a model")
    versions_parser.add_argument(
        "--model",
        choices=["linear_regression", "lightgbm"],
        required=True,
    )
    versions_parser.add_argument(
        "--handler",
        choices=["azure", "local"],
        default=settings.default_model_handler,
    )
    versions_parser.set_defaults(func=handle_versions)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
