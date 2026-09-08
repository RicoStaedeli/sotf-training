from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class WandbSettings:
    project: str | None = None
    group: str | None = None
    api_key_file: str | None = None
    run_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Enabled.

        Returns:
            bool: Result of the function.
        """
        return bool(self.project)


class WandbRun:
    def __init__(self, settings: WandbSettings) -> None:
        """Initialize instance.

        Args:
            settings (WandbSettings): WandbSettings value used by the function.

        Returns:
            None: None.
        """
        self._settings = settings
        self._run: Any | None = None

    def __enter__(self) -> "WandbRun":
        """Enter.

        Returns:
            "WandbRun": Result of the function.
        """
        if not self._settings.enabled:
            return self

        if self._settings.api_key_file:
            key_path = Path(self._settings.api_key_file).expanduser()
            if not key_path.exists():
                log.warning("W&B API key file does not exist: %s. Skipping W&B logging.", key_path)
                return self
            api_key = key_path.read_text(encoding="utf-8").strip()
            if not api_key:
                log.warning("W&B API key file is empty: %s. Skipping W&B logging.", key_path)
                return self
            os.environ["WANDB_API_KEY"] = api_key

        try:
            import wandb
        except ImportError:
            log.warning("wandb is not installed. Skipping W&B logging.")
            return self

        self._run = wandb.init(
            project=self._settings.project,
            group=self._settings.group,
            name=self._settings.run_name,
            config=self._settings.config or None,
            reinit=True,
        )
        log.info("W&B run initialised: project=%s group=%s", self._settings.project, self._settings.group)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Exit.

        Args:
            exc_type (Any): Value used by the function.
            exc (Any): Value used by the function.
            traceback (Any): Value used by the function.

        Returns:
            None: None.
        """
        if self._run is not None:
            self._run.finish()
            self._run = None

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Log.

        Args:
            metrics (dict[str, Any]): dict[str, Any] value used by the function.
            step (int | None): int | None value used by the function.

        Returns:
            None: None.
        """
        if self._run is None:
            return
        self._run.log(metrics, step=step)
