import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WandbLogger:
    """Logs scalar metrics to Weights & Biases when configured."""

    def __init__(
        self,
        log_dir: Path,
        project: str,
        group: str | None = None,
        api_key_file: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        self._run = None

        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not available. Install wandb to enable logging.")
            return

        key = _read_wandb_api_key(api_key_file)
        if key:
            try:
                wandb.login(key=key)
            except Exception as exc:
                logger.warning("Failed to log in to wandb using %s: %s", api_key_file, exc)

        try:
            self._run = wandb.init(
                project=project,
                group=group,
                dir=str(log_dir),
                config=config,
                name=run_name,
                reinit=True,
            )
        except Exception as exc:
            logger.warning("Failed to initialize wandb logging: %s", exc)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._run is None:
            return
        self._run.log(self._flatten_metrics(metrics), step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

    def _flatten_metrics(
        self,
        metrics: Mapping[str, Any],
        prefix: str = "",
    ) -> dict[str, float]:
        items: dict[str, float] = {}

        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key

            if isinstance(value, Mapping):
                items.update(self._flatten_metrics(value, tag))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                items[tag] = float(value)

        return items


def _read_wandb_api_key(api_key_file: str | None) -> str | None:
    if not api_key_file:
        return None

    try:
        key = Path(api_key_file).read_text().strip()
    except OSError as exc:
        logger.warning("Failed to read wandb API key file %s: %s", api_key_file, exc)
        return None

    return key or None
