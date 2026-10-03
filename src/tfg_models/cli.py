"""Command Line Interface for training, evaluating, and predicting property models."""

import argparse
import logging
import sys
from typing import Optional

from tfg_models.config import settings
from tfg_models.core.model_handler import ModelHandler, get_model_handler
from tfg_models.models import MODEL_REGISTRY, get_model_trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_model_handler(model_name: str) -> ModelHandler:
    return get_model_handler(model_name)


def cmd_train(args: argparse.Namespace) -> None:
    """Handles the 'train' subcommand."""
    targets = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    print("\n" + "=" * 60)
    print(f"TRAINING PIPELINE: {targets}")
    print("=" * 60 + "\n")

    for model_name in targets:
        print(f"\n--- Running pipeline for: {model_name} ---")
        trainer = get_model_trainer(model_name)
        _, _, report, version_id = trainer.run(version=args.version)

        metrics = report.get("metrics") or report
        print(f"\nSuccessfully trained {model_name} (version: {version_id})")
        print(f"  MAE:  {metrics.get('mae', 0.0):,.2f} €")
        print(f"  RMSE: {metrics.get('rmse', 0.0):,.2f} €")
        print(f"  R²:   {metrics.get('r2_score', 0.0):.4f}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Handles the 'predict' subcommand."""
    trainer = get_model_trainer(args.model)
    version = args.version or "latest"

    price = trainer.predict_sample(
        surface=args.surface,
        rooms=args.rooms,
        bathrooms=args.bathrooms,
        postal_code=str(args.postal_code),
        elevator=args.elevator,
        terrace=args.terrace,
        garage=args.garage,
        version=version,
    )

    print("\n" + "=" * 50)
    print(f"PROPERTY VALUATION RESULT ({args.model.upper()})")
    print("=" * 50)
    print(f"  Postal Code: {args.postal_code}")
    print(f"  Surface:     {args.surface:g} m²")
    print(f"  Rooms:       {args.rooms} | Bathrooms: {args.bathrooms}")
    print(
        f"  Features:    Elevator={args.elevator}, Terrace={args.terrace}, Garage={args.garage}"
    )
    print("-" * 50)
    print(f"  ESTIMATED PRICE: {price:,.2f} €")
    print("=" * 50 + "\n")


def cmd_compare(args: argparse.Namespace) -> None:
    """Handles the 'compare' subcommand, displaying metrics for all registered models."""
    print("\n" + "=" * 70)
    print(
        f"{'MODEL':<20} | {'MAE (€)':<12} | {'RMSE (€)':<12} | {'R²':<8} | {'VERSION':<15}"
    )
    print("-" * 70)

    for model_name in MODEL_REGISTRY:
        try:
            handler = _get_model_handler(model_name)
            report = handler.get_report(version="latest")
            metrics = report.get("metrics") or report
            mae = metrics.get("mae", 0.0)
            rmse = metrics.get("rmse", 0.0)
            r2 = metrics.get("r2_score", 0.0)
            version = report.get("version", "latest")
            print(
                f"{model_name:<20} | {mae:<12,.2f} | {rmse:<12,.2f} | {r2:<8.4f} | {version:<15}"
            )
        except Exception as e:
            print(f"{model_name:<20} | {'(no trained model found)':<45}")

    print("=" * 70 + "\n")


def cmd_versions(args: argparse.Namespace) -> None:
    """Handles the 'versions' subcommand."""
    try:
        handler = _get_model_handler(args.model)
        versions = handler.list_versions()
    except Exception as e:
        print(f"Error accessing storage: {e}")
        return

    print("\n" + "=" * 40)
    print(f"SAVED VERSIONS FOR '{args.model}'")
    print("=" * 40)
    if not versions:
        print("  (no versions found)")
    else:
        for v in versions:
            print(f"  - {v}")
    print("=" * 40 + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Constructs the root CLI parser and subcommands."""
    parser = argparse.ArgumentParser(
        description="TFG Real Estate Valuation Models CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train property valuation models")
    train_parser.add_argument(
        "--model",
        choices=["all"] + list(MODEL_REGISTRY.keys()),
        default="all",
        help="Model to train or 'all'",
    )
    train_parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Optional custom version ID (default: YYYYMMDD_HHMMSS)",
    )
    train_parser.set_defaults(func=cmd_train)

    # predict
    predict_parser = subparsers.add_parser("predict", help="Predict property valuation")
    predict_parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="lightgbm",
        help="Model to use for prediction",
    )
    predict_parser.add_argument(
        "--surface", type=float, required=True, help="Surface area in m²"
    )
    predict_parser.add_argument("--rooms", type=int, required=True, help="Number of rooms")
    predict_parser.add_argument(
        "--bathrooms", type=int, required=True, help="Number of bathrooms"
    )
    predict_parser.add_argument(
        "--postal-code", type=str, required=True, help="Postal code (e.g. 36211)"
    )
    predict_parser.add_argument(
        "--elevator", action="store_true", help="Has elevator"
    )
    predict_parser.add_argument(
        "--terrace", action="store_true", help="Has terrace"
    )
    predict_parser.add_argument(
        "--garage", action="store_true", help="Has garage"
    )
    predict_parser.add_argument(
        "--version", type=str, default="latest", help="Model version to use"
    )
    predict_parser.set_defaults(func=cmd_predict)

    # compare
    compare_parser = subparsers.add_parser(
        "compare", help="Compare performance of trained models"
    )
    compare_parser.set_defaults(func=cmd_compare)

    # versions
    versions_parser = subparsers.add_parser(
        "versions", help="List saved versions for a model"
    )
    versions_parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help="Model name",
    )
    versions_parser.set_defaults(func=cmd_versions)

    return parser


def main(argv: Optional[list] = None) -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
