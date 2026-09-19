"""Run logging: a CSV of scalar metrics, optional TensorBoard scalars and console lines.

Layout inside the run directory: `metrics.csv`, `log.txt` (copy of the console
lines) and `tb/` (TensorBoard event files, only when enabled).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

CSV_NAME = "metrics.csv"
LOG_NAME = "log.txt"
TB_DIR = "tb"
STEP_COLUMN = "step"


class Logger:
    """Writes scalar metrics of one training run.

    `metrics.csv` has a `step` column followed by every metric key in order of
    first appearance. Different `log` calls may carry different keys (rollout
    stats, losses, eval results): when a new key shows up the file is rewritten
    with the wider header and earlier rows are padded with empty cells, so the CSV
    is rectangular and loadable at every moment. Each row is flushed immediately.

    With `resume=True` an existing CSV (and `log.txt`) is appended to instead of
    replaced. `resume_step`, if given, first drops rows logged *after* that step -
    the ones a crashed run wrote between its last checkpoint and the crash, which
    the resumed run is about to log again - and purges the same range from TensorBoard.

    TensorBoard (`torch.utils.tensorboard`) is imported only when `use_tensorboard`
    is true, so CSV-only logging never pays for that import.
    """

    def __init__(
        self,
        run_dir: str | Path,
        use_tensorboard: bool = True,
        resume: bool = False,
        *,
        resume_step: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / CSV_NAME
        self._closed = False
        self._csv_file = None
        self._log_file = None
        self._tb = None

        self._keys: list[str] = []
        self._header_written = False
        if resume and self.csv_path.is_file():
            self._header_written = self._adopt_existing_csv(resume_step)
        # Both files stay open for the lifetime of the run; rows are flushed one by one.
        csv_mode = "a" if self._header_written else "w"
        self._csv_file = open(self.csv_path, csv_mode, newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._log_file = open(self.run_dir / LOG_NAME, "a" if resume else "w", encoding="utf-8")

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                self.close()
                raise ImportError(
                    "TensorBoard logging needs the `tensorboard` package "
                    "(install it, or set tensorboard=false in the config)"
                ) from exc
            purge_step = resume_step + 1 if resume and resume_step is not None else None
            self._tb = SummaryWriter(log_dir=str(self.run_dir / TB_DIR), purge_step=purge_step)

    # ------------------------------------------------------------------ public

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record one row of scalar `metrics` at env-transition count `step`.

        Values may be Python or numpy scalars; `None` values are skipped (empty CSV
        cell, no TensorBoard point).
        """
        if self._closed:
            raise RuntimeError("Logger is closed")
        if STEP_COLUMN in metrics:
            raise ValueError(f"{STEP_COLUMN!r} is a reserved column; pass it as the step argument")
        step = int(step)
        values = {str(key): value for key, value in metrics.items() if value is not None}

        new_keys = [str(key) for key in metrics if str(key) not in self._keys]
        if new_keys or not self._header_written:
            self._widen_header(new_keys)
        row = [str(step)] + [_format(values[key]) if key in values else "" for key in self._keys]
        self._writer.writerow(row)
        self._csv_file.flush()

        if self._tb is not None:
            for key, value in values.items():
                self._tb.add_scalar(key, float(value), global_step=step)

    def print(self, line: str) -> None:
        """Print a console line (flushed) and append it to `log.txt`."""
        print(line, flush=True)
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.write(line + "\n")
            self._log_file.flush()

    def flush(self) -> None:
        """Push buffered TensorBoard events to disk (CSV rows are always flushed)."""
        if self._tb is not None:
            self._tb.flush()

    def close(self) -> None:
        """Flush and close every sink. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._tb is not None:
            self._tb.close()
        for handle in (self._csv_file, self._log_file):
            if handle is not None and not handle.closed:
                handle.close()

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ----------------------------------------------------------------- helpers

    def _adopt_existing_csv(self, resume_step: int | None) -> bool:
        """Take over the CSV being resumed; returns False if it is unusable (empty/foreign).

        Rows beyond `resume_step` are dropped.
        """
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows or not rows[0] or rows[0][0] != STEP_COLUMN:
            return False
        header, body = rows[0], rows[1:]
        if resume_step is not None:
            kept = [row for row in body if _row_step(row) <= resume_step]
            if len(kept) != len(body):
                self._rewrite(header, kept)
        self._keys = header[1:]
        return True

    def _widen_header(self, new_keys: list[str]) -> None:
        """Add columns for `new_keys`, rewriting the rows already on disk."""
        self._csv_file.close()
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as fh:
                old_rows = list(csv.reader(fh))[1:]
            keys = [*self._keys, *new_keys]
            width = 1 + len(keys)
            padded = [row + [""] * (width - len(row)) for row in old_rows]
            self._rewrite([STEP_COLUMN, *keys], padded)
            self._keys = keys
            self._header_written = True
        finally:  # whatever happened, keep the logger usable
            self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._csv_file)

    def _rewrite(self, header: list[str], rows: list[list[str]]) -> None:
        """Replace the CSV atomically, so an interruption never leaves half a file."""
        tmp = self.csv_path.with_name(f".{CSV_NAME}.tmp")
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(rows)
            os.replace(tmp, self.csv_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def _row_step(row: list[str]) -> float:
    try:
        return float(row[0])
    except (IndexError, ValueError):
        return float("inf")  # malformed row: treat as "after the checkpoint" and drop it


def _format(value: Any) -> str:
    """Compact text for a CSV cell: integers exactly, floats with 8 significant digits."""
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.8g}"
