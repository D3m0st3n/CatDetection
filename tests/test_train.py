"""Unit tests for src.train."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.train import (
    DEFAULT_BASE_MODEL,
    STAGE1_EPOCHS,
    STAGE1_FREEZE,
    STAGE1_LR0,
    STAGE2_EPOCHS,
    STAGE2_LR0,
    TrainingResult,
    find_latest_run_dir,
    run_stage1,
    run_stage2,
    run_training,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_run_dir(
    project_dir: Path,
    name: str = "train",
) -> Path:
    """Create a fake YOLOv8 run directory with dummy weight files.

    Args:
        project_dir: Parent directory (e.g. ``runs/detect``).
        name: Run directory name (e.g. ``"train"``, ``"train2"``).

    Returns:
        Path to the created run directory.
    """
    run_dir = project_dir / name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_text("fake")
    (weights_dir / "last.pt").write_text("fake")
    (run_dir / "results.csv").write_text("epoch,mAP\n1,0.5\n")
    return run_dir


def _create_data_yaml(tmp_path: Path) -> Path:
    """Create a minimal data YAML file for testing.

    Args:
        tmp_path: Temporary directory.

    Returns:
        Path to the created YAML file.
    """
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("nc: 2\nnames: ['aioli', 'mayo']\n")
    return data_yaml


# ---------------------------------------------------------------------------
# find_latest_run_dir
# ---------------------------------------------------------------------------


class TestFindLatestRunDir:
    """Tests for find_latest_run_dir."""

    def test_single_run_dir(self, tmp_path: Path) -> None:
        """A single train/ directory is found and returned."""
        project = tmp_path / "detect"
        _create_fake_run_dir(project, "train")

        result = find_latest_run_dir(project)
        assert result.name == "train"

    def test_multiple_run_dirs(self, tmp_path: Path) -> None:
        """With train/, train2/, train3/, returns train3/."""
        project = tmp_path / "detect"
        _create_fake_run_dir(project, "train")
        _create_fake_run_dir(project, "train2")
        _create_fake_run_dir(project, "train3")

        result = find_latest_run_dir(project)
        assert result.name == "train3"

    def test_nonconsecutive_numbering(self, tmp_path: Path) -> None:
        """With train/ and train5/, returns train5/."""
        project = tmp_path / "detect"
        _create_fake_run_dir(project, "train")
        _create_fake_run_dir(project, "train5")

        result = find_latest_run_dir(project)
        assert result.name == "train5"

    def test_ignores_non_train_dirs(self, tmp_path: Path) -> None:
        """Directories like predict/ or val/ are ignored."""
        project = tmp_path / "detect"
        _create_fake_run_dir(project, "train")
        (project / "predict").mkdir(parents=True)
        (project / "val").mkdir(parents=True)
        (project / "training_notes").mkdir(parents=True)

        result = find_latest_run_dir(project)
        assert result.name == "train"

    def test_raises_when_no_runs(self, tmp_path: Path) -> None:
        """Empty project dir raises FileNotFoundError."""
        project = tmp_path / "detect"
        project.mkdir()

        with pytest.raises(FileNotFoundError, match="No train"):
            find_latest_run_dir(project)

    def test_raises_when_project_missing(self, tmp_path: Path) -> None:
        """Non-existent project dir raises FileNotFoundError."""
        project = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Project directory"):
            find_latest_run_dir(project)


# ---------------------------------------------------------------------------
# run_stage1
# ---------------------------------------------------------------------------


class TestRunStage1:
    """Tests for run_stage1."""

    @patch("src.train.find_latest_run_dir")
    @patch("src.train.YOLO")
    def test_calls_yolo_with_correct_defaults(
        self, mock_yolo_cls: MagicMock, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        """YOLO is called with the default base model and correct train kwargs."""
        data_yaml = _create_data_yaml(tmp_path)
        fake_run = _create_fake_run_dir(tmp_path / "runs" / "detect")
        mock_find.return_value = fake_run
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        run_stage1(
            data_yaml=data_yaml,
            project=str(tmp_path / "runs" / "detect"),
        )

        mock_yolo_cls.assert_called_once_with(DEFAULT_BASE_MODEL)
        mock_model.train.assert_called_once()
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs["epochs"] == STAGE1_EPOCHS
        assert call_kwargs["freeze"] == STAGE1_FREEZE
        assert call_kwargs["lr0"] == STAGE1_LR0

    @patch("src.train.find_latest_run_dir")
    @patch("src.train.YOLO")
    def test_custom_hyperparameters_passed(
        self, mock_yolo_cls: MagicMock, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        """Overridden values reach model.train()."""
        data_yaml = _create_data_yaml(tmp_path)
        fake_run = _create_fake_run_dir(tmp_path / "runs" / "detect")
        mock_find.return_value = fake_run
        mock_yolo_cls.return_value = MagicMock()

        run_stage1(
            data_yaml=data_yaml,
            epochs=10,
            freeze=5,
            lr0=0.005,
            device="cpu",
            project=str(tmp_path / "runs" / "detect"),
        )

        call_kwargs = mock_yolo_cls.return_value.train.call_args[1]
        assert call_kwargs["epochs"] == 10
        assert call_kwargs["freeze"] == 5
        assert call_kwargs["lr0"] == 0.005
        assert call_kwargs["device"] == "cpu"

    @patch("src.train.find_latest_run_dir")
    @patch("src.train.YOLO")
    def test_returns_run_dir(
        self, mock_yolo_cls: MagicMock, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        """The return value is the run dir containing weights/last.pt."""
        data_yaml = _create_data_yaml(tmp_path)
        fake_run = _create_fake_run_dir(tmp_path / "runs" / "detect")
        mock_find.return_value = fake_run
        mock_yolo_cls.return_value = MagicMock()

        result = run_stage1(
            data_yaml=data_yaml,
            project=str(tmp_path / "runs" / "detect"),
        )

        assert result == fake_run
        assert (result / "weights" / "last.pt").exists()

    def test_raises_on_missing_data_yaml(self, tmp_path: Path) -> None:
        """A non-existent data_yaml raises FileNotFoundError before YOLO."""
        missing = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Data YAML"):
            run_stage1(data_yaml=missing)

    @patch("src.train.find_latest_run_dir")
    @patch("src.train.YOLO")
    def test_raises_on_missing_weights(
        self, mock_yolo_cls: MagicMock, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        """RuntimeError when the run dir has no weights/last.pt."""
        data_yaml = _create_data_yaml(tmp_path)
        # Create a run dir without weights.
        empty_run = tmp_path / "runs" / "detect" / "train"
        empty_run.mkdir(parents=True)
        mock_find.return_value = empty_run
        mock_yolo_cls.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="did not produce weights"):
            run_stage1(
                data_yaml=data_yaml,
                project=str(tmp_path / "runs" / "detect"),
            )


# ---------------------------------------------------------------------------
# run_stage2
# ---------------------------------------------------------------------------


class TestRunStage2:
    """Tests for run_stage2."""

    @patch("src.train.find_latest_run_dir")
    @patch("src.train.YOLO")
    def test_loads_checkpoint_from_stage1(
        self, mock_yolo_cls: MagicMock, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        """YOLO is called with stage1_dir/weights/last.pt."""
        data_yaml = _create_data_yaml(tmp_path)
        stage1_dir = _create_fake_run_dir(tmp_path / "runs" / "detect", "train")
        stage2_dir = _create_fake_run_dir(tmp_path / "runs" / "detect", "train2")
        mock_find.return_value = stage2_dir
        mock_yolo_cls.return_value = MagicMock()

        run_stage2(
            stage1_dir=stage1_dir,
            data_yaml=data_yaml,
            project=str(tmp_path / "runs" / "detect"),
        )

        expected_checkpoint = str(stage1_dir / "weights" / "last.pt")
        mock_yolo_cls.assert_called_once_with(expected_checkpoint)

    @patch("src.train.find_latest_run_dir")
    @patch("src.train.YOLO")
    def test_calls_train_with_stage2_defaults(
        self, mock_yolo_cls: MagicMock, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        """model.train() gets correct epochs/lr and no freeze argument."""
        data_yaml = _create_data_yaml(tmp_path)
        stage1_dir = _create_fake_run_dir(tmp_path / "runs" / "detect", "train")
        stage2_dir = _create_fake_run_dir(tmp_path / "runs" / "detect", "train2")
        mock_find.return_value = stage2_dir
        mock_yolo_cls.return_value = MagicMock()

        run_stage2(
            stage1_dir=stage1_dir,
            data_yaml=data_yaml,
            project=str(tmp_path / "runs" / "detect"),
        )

        call_kwargs = mock_yolo_cls.return_value.train.call_args[1]
        assert call_kwargs["epochs"] == STAGE2_EPOCHS
        assert call_kwargs["lr0"] == STAGE2_LR0
        assert "freeze" not in call_kwargs

    @patch("src.train.find_latest_run_dir")
    @patch("src.train.YOLO")
    def test_returns_run_dir(
        self, mock_yolo_cls: MagicMock, mock_find: MagicMock, tmp_path: Path
    ) -> None:
        """Return value is the Stage 2 run dir with weights/best.pt."""
        data_yaml = _create_data_yaml(tmp_path)
        stage1_dir = _create_fake_run_dir(tmp_path / "runs" / "detect", "train")
        stage2_dir = _create_fake_run_dir(tmp_path / "runs" / "detect", "train2")
        mock_find.return_value = stage2_dir
        mock_yolo_cls.return_value = MagicMock()

        result = run_stage2(
            stage1_dir=stage1_dir,
            data_yaml=data_yaml,
            project=str(tmp_path / "runs" / "detect"),
        )

        assert result == stage2_dir
        assert (result / "weights" / "best.pt").exists()

    def test_raises_on_missing_checkpoint(self, tmp_path: Path) -> None:
        """FileNotFoundError when stage1_dir/weights/last.pt is missing."""
        data_yaml = _create_data_yaml(tmp_path)
        empty_stage1 = tmp_path / "runs" / "detect" / "train"
        empty_stage1.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="checkpoint"):
            run_stage2(stage1_dir=empty_stage1, data_yaml=data_yaml)

    def test_raises_on_missing_data_yaml(self, tmp_path: Path) -> None:
        """FileNotFoundError when data YAML does not exist."""
        stage1_dir = _create_fake_run_dir(tmp_path / "runs" / "detect", "train")
        missing = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Data YAML"):
            run_stage2(stage1_dir=stage1_dir, data_yaml=missing)


# ---------------------------------------------------------------------------
# run_training
# ---------------------------------------------------------------------------


class TestRunTraining:
    """Tests for run_training (orchestration)."""

    @patch("src.train.run_stage2")
    @patch("src.train.run_stage1")
    def test_calls_both_stages_in_order(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Stage 1 runs first; Stage 2 receives stage1_dir."""
        data_yaml = _create_data_yaml(tmp_path)
        fake_s1 = _create_fake_run_dir(tmp_path / "runs" / "detect", "train")
        fake_s2 = _create_fake_run_dir(tmp_path / "runs" / "detect", "train2")
        mock_stage1.return_value = fake_s1
        mock_stage2.return_value = fake_s2

        run_training(data_yaml=data_yaml, project=str(tmp_path / "runs" / "detect"))

        mock_stage1.assert_called_once()
        mock_stage2.assert_called_once()
        assert mock_stage2.call_args[1]["stage1_dir"] == fake_s1

    @patch("src.train.run_stage2")
    @patch("src.train.run_stage1")
    def test_returns_training_result(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """All TrainingResult fields are populated."""
        data_yaml = _create_data_yaml(tmp_path)
        fake_s1 = _create_fake_run_dir(tmp_path / "runs" / "detect", "train")
        fake_s2 = _create_fake_run_dir(tmp_path / "runs" / "detect", "train2")
        mock_stage1.return_value = fake_s1
        mock_stage2.return_value = fake_s2

        result = run_training(
            data_yaml=data_yaml, project=str(tmp_path / "runs" / "detect")
        )

        assert isinstance(result, TrainingResult)
        assert result.stage1_dir == fake_s1
        assert result.stage2_dir == fake_s2
        assert result.best_weights == fake_s2 / "weights" / "best.pt"
        assert result.last_weights == fake_s2 / "weights" / "last.pt"
        assert result.results_csv == fake_s2 / "results.csv"

    @patch("src.train.run_stage2")
    @patch("src.train.run_stage1")
    def test_stage1_failure_prevents_stage2(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """If Stage 1 raises, Stage 2 is never called."""
        data_yaml = _create_data_yaml(tmp_path)
        mock_stage1.side_effect = RuntimeError("Stage 1 failed")

        with pytest.raises(RuntimeError, match="Stage 1 failed"):
            run_training(
                data_yaml=data_yaml,
                project=str(tmp_path / "runs" / "detect"),
            )

        mock_stage2.assert_not_called()

    @patch("src.train.run_stage2")
    @patch("src.train.run_stage1")
    def test_custom_params_forwarded(
        self, mock_stage1: MagicMock, mock_stage2: MagicMock, tmp_path: Path
    ) -> None:
        """Custom epochs/lr values reach the correct stage functions."""
        data_yaml = _create_data_yaml(tmp_path)
        fake_s1 = _create_fake_run_dir(tmp_path / "runs" / "detect", "train")
        fake_s2 = _create_fake_run_dir(tmp_path / "runs" / "detect", "train2")
        mock_stage1.return_value = fake_s1
        mock_stage2.return_value = fake_s2

        run_training(
            data_yaml=data_yaml,
            stage1_epochs=10,
            stage1_lr0=0.005,
            stage2_epochs=30,
            stage2_lr0=0.0005,
            project=str(tmp_path / "runs" / "detect"),
        )

        s1_kwargs = mock_stage1.call_args[1]
        assert s1_kwargs["epochs"] == 10
        assert s1_kwargs["lr0"] == 0.005

        s2_kwargs = mock_stage2.call_args[1]
        assert s2_kwargs["epochs"] == 30
        assert s2_kwargs["lr0"] == 0.0005
