"""Unit tests for the CLI command interface."""

import pytest
from unittest.mock import MagicMock, patch

from tfg_models.cli import build_parser, main


def test_cli_help():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


@patch("tfg_models.cli.get_model_trainer")
def test_cmd_train(mock_get_trainer, capsys):
    mock_trainer = MagicMock()
    mock_trainer.run.return_value = (
        None,
        ["surface"],
        {"mae": 15000.0, "rmse": 25000.0, "r2_score": 0.85},
        "v_test",
    )
    mock_get_trainer.return_value = mock_trainer

    main(["train", "--model", "lightgbm", "--version", "v_test"])

    out, _ = capsys.readouterr()
    assert "TRAINING PIPELINE: ['lightgbm']" in out
    assert "Successfully trained lightgbm (version: v_test)" in out
    assert "MAE:  15,000.00 €" in out


@patch("tfg_models.cli.get_model_trainer")
def test_cmd_predict(mock_get_trainer, capsys):
    mock_trainer = MagicMock()
    mock_trainer.predict_sample.return_value = 285000.50
    mock_get_trainer.return_value = mock_trainer

    main(
        [
            "predict",
            "--model",
            "lightgbm",
            "--surface",
            "85",
            "--rooms",
            "3",
            "--bathrooms",
            "2",
            "--postal-code",
            "36211",
            "--elevator",
            "--terrace",
        ]
    )

    out, _ = capsys.readouterr()
    assert "PROPERTY VALUATION RESULT (LIGHTGBM)" in out
    assert "ESTIMATED PRICE: 285,000.50 €" in out


@patch("tfg_models.cli.get_model_handler")
def test_cmd_compare(mock_get_handler, capsys):
    mock_handler = MagicMock()
    mock_handler.get_report.return_value = {
        "mae": 75000.0,
        "rmse": 140000.0,
        "r2_score": 0.58,
        "version": "20260930_120000",
    }
    mock_get_handler.return_value = mock_handler

    main(["compare"])

    out, _ = capsys.readouterr()
    assert "MODEL" in out
    assert "MAE (€)" in out
    assert "75,000.00" in out


@patch("tfg_models.cli.get_model_handler")
def test_cmd_versions(mock_get_handler, capsys):
    mock_handler = MagicMock()
    mock_handler.list_versions.return_value = ["20260930_120000", "20260929_100000"]
    mock_get_handler.return_value = mock_handler

    main(["versions", "--model", "lightgbm"])

    out, _ = capsys.readouterr()
    assert "SAVED VERSIONS FOR 'lightgbm'" in out
    assert "20260930_120000" in out
    assert "20260929_100000" in out
