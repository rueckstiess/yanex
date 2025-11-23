"""Format handlers for artifact saving and loading with automatic detection."""

import csv
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FormatHandler:
    """Handler for a specific artifact format."""

    extensions: list[str]
    type_check: Callable[[Any], bool]
    saver: Callable[[Any, Path], None]
    loader: Callable[[Path], Any]
    required_package: str | None = None


def _save_text(obj: Any, path: Path) -> None:
    """Save string as text file."""
    if not isinstance(obj, str):
        raise TypeError(f"Expected str for .txt file, got {type(obj).__name__}")
    path.write_text(obj, encoding="utf-8")


def _load_text(path: Path) -> str:
    """Load text file as string."""
    return path.read_text(encoding="utf-8")


def _save_json(obj: Any, path: Path) -> None:
    """Save object as JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: Path) -> Any:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_jsonl(obj: Any, path: Path) -> None:
    """Save list of objects as JSON Lines."""
    if not isinstance(obj, list):
        raise TypeError(f"Expected list for .jsonl file, got {type(obj).__name__}")
    with path.open("w", encoding="utf-8") as f:
        for item in obj:
            f.write(json.dumps(item) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines file as list of dicts."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _save_csv_pandas(obj: Any, path: Path) -> None:
    """Save pandas DataFrame as CSV."""
    import pandas as pd

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame for .csv file, got {type(obj).__name__}"
        )
    obj.to_csv(path, index=False)


def _load_csv_pandas(path: Path) -> Any:
    """Load CSV file as pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(path)


def _save_csv_list(obj: Any, path: Path) -> None:
    """Save list of dicts as CSV."""
    if not isinstance(obj, list):
        raise TypeError(f"Expected list for .csv file, got {type(obj).__name__}")
    if not obj:
        # Empty list - create empty CSV
        path.write_text("", encoding="utf-8")
        return
    if not isinstance(obj[0], dict):
        raise TypeError(
            f"Expected list of dicts for .csv file, got list of {type(obj[0]).__name__}"
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=obj[0].keys())
        writer.writeheader()
        writer.writerows(obj)


def _save_npy(obj: Any, path: Path) -> None:
    """Save numpy array as .npy."""
    import numpy as np

    if not isinstance(obj, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray for .npy file, got {type(obj).__name__}"
        )
    np.save(path, obj)


def _load_npy(path: Path) -> Any:
    """Load .npy file as numpy array."""
    import numpy as np

    return np.load(path)


def _save_npz(obj: Any, path: Path) -> None:
    """Save dict of numpy arrays as .npz."""
    import numpy as np

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict for .npz file, got {type(obj).__name__}")
    # Verify all values are numpy arrays
    for key, value in obj.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Expected dict of numpy arrays for .npz file, got dict with {type(value).__name__} at key '{key}'"
            )
    np.savez(path, **obj)


def _load_npz(path: Path) -> dict[str, Any]:
    """Load .npz file as dict of numpy arrays."""
    import numpy as np

    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


def _save_torch(obj: Any, path: Path) -> None:
    """Save object with torch.save."""
    import torch

    torch.save(obj, path)


def _load_torch(path: Path) -> Any:
    """Load object with torch.load.

    Note: Uses weights_only=True by default for security. If you need to load
    arbitrary Python objects, use a custom loader with weights_only=False.
    """
    import torch

    # Use weights_only=True for security (prevents arbitrary code execution)
    # This is safe for model weights but not for arbitrary pickled objects
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        # If weights_only=True fails, it might be an older checkpoint format
        # Fall back to weights_only=False but warn the user
        import warnings

        warnings.warn(
            f"Loading {path.name} with weights_only=False. "
            "This may execute arbitrary code. "
            "Consider re-saving with a newer PyTorch version.",
            UserWarning,
            stacklevel=2,
        )
        return torch.load(path, weights_only=False)


def _save_pickle(obj: Any, path: Path) -> None:
    """Save object with pickle."""
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path) -> Any:
    """Load object with pickle."""
    with path.open("rb") as f:
        return pickle.load(f)


def _save_matplotlib_figure(obj: Any, path: Path) -> None:
    """Save matplotlib figure as image."""
    # Check if it's a matplotlib figure
    try:
        import matplotlib.figure
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for saving .png figures. "
            "Install it with: pip install matplotlib"
        ) from err

    if not isinstance(obj, matplotlib.figure.Figure):
        raise TypeError(
            f"Expected matplotlib.figure.Figure for .png file, got {type(obj).__name__}"
        )
    obj.savefig(path)


def _load_image(path: Path) -> Any:
    """Load image file with PIL."""
    try:
        from PIL import Image
    except ImportError as err:
        raise ImportError(
            "Pillow is required for loading image files. "
            "Install it with: pip install Pillow"
        ) from err

    return Image.open(path)


# Registry of format handlers
# Order matters: more specific handlers should come first
FORMAT_HANDLERS = [
    # Text
    FormatHandler(
        extensions=[".txt"],
        type_check=lambda obj: isinstance(obj, str),
        saver=_save_text,
        loader=_load_text,
    ),
    # CSV with pandas
    FormatHandler(
        extensions=[".csv"],
        type_check=lambda obj: type(obj).__name__ == "DataFrame",
        saver=_save_csv_pandas,
        loader=_load_csv_pandas,
        required_package="pandas",
    ),
    # CSV with list of dicts
    FormatHandler(
        extensions=[".csv"],
        type_check=lambda obj: isinstance(obj, list)
        and (not obj or isinstance(obj[0], dict)),
        saver=_save_csv_list,
        loader=_load_csv_pandas,  # Load as pandas by default
        required_package="pandas",
    ),
    # JSON
    FormatHandler(
        extensions=[".json"],
        type_check=lambda obj: True,  # JSON handles many types
        saver=_save_json,
        loader=_load_json,
    ),
    # JSON Lines
    FormatHandler(
        extensions=[".jsonl"],
        type_check=lambda obj: isinstance(obj, list),
        saver=_save_jsonl,
        loader=_load_jsonl,
    ),
    # NumPy array
    FormatHandler(
        extensions=[".npy"],
        type_check=lambda obj: type(obj).__name__ == "ndarray",
        saver=_save_npy,
        loader=_load_npy,
        required_package="numpy",
    ),
    # NumPy arrays dict
    FormatHandler(
        extensions=[".npz"],
        type_check=lambda obj: isinstance(obj, dict)
        and all(type(v).__name__ == "ndarray" for v in obj.values()),
        saver=_save_npz,
        loader=_load_npz,
        required_package="numpy",
    ),
    # PyTorch
    FormatHandler(
        extensions=[".pt", ".pth"],
        type_check=lambda obj: True,  # torch.save handles any object
        saver=_save_torch,
        loader=_load_torch,
        required_package="torch",
    ),
    # Pickle (fallback for any object)
    FormatHandler(
        extensions=[".pkl"],
        type_check=lambda obj: True,  # Pickle handles any object
        saver=_save_pickle,
        loader=_load_pickle,
    ),
    # Matplotlib figure / PNG image
    FormatHandler(
        extensions=[".png"],
        type_check=lambda obj: type(obj).__name__ == "Figure",
        saver=_save_matplotlib_figure,
        loader=_load_image,
        required_package="matplotlib",
    ),
]


def get_handler_for_save(obj: Any, filename: str) -> FormatHandler:
    """Find appropriate handler for saving based on object type and filename.

    Args:
        obj: Object to save
        filename: Target filename (extension determines format)

    Returns:
        FormatHandler for the object and filename

    Raises:
        ValueError: If no handler found for the extension
        TypeError: If object type doesn't match the expected type for the extension
    """
    ext = Path(filename).suffix.lower()

    # Find handlers that match the extension
    matching_handlers = [h for h in FORMAT_HANDLERS if ext in h.extensions]

    if not matching_handlers:
        supported = sorted({ext for h in FORMAT_HANDLERS for ext in h.extensions})
        raise ValueError(
            f"Cannot auto-detect format for '{filename}'. "
            f"Supported extensions: {', '.join(supported)}. "
            f"Use 'saver' parameter for custom formats."
        )

    # Try each matching handler
    for handler in matching_handlers:
        if handler.type_check(obj):
            # Check if required package is available
            if handler.required_package:
                try:
                    __import__(handler.required_package)
                except ImportError as err:
                    raise ImportError(
                        f"Saving {filename} requires {handler.required_package}. "
                        f"Install with: pip install {handler.required_package}"
                    ) from err
            return handler

    # No handler matched the object type
    raise TypeError(
        f"Cannot save object of type {type(obj).__name__} to {filename}. "
        f"Expected one of the supported types for {ext} files."
    )


def get_handler_for_load(filename: str) -> FormatHandler:
    """Find appropriate handler for loading based on filename extension.

    Args:
        filename: Filename to load (extension determines format)

    Returns:
        FormatHandler for the filename

    Raises:
        ValueError: If no handler found for the extension
        ImportError: If required package is not installed
    """
    ext = Path(filename).suffix.lower()

    # Find first handler that matches the extension
    for handler in FORMAT_HANDLERS:
        if ext in handler.extensions:
            # Check if required package is available
            if handler.required_package:
                try:
                    __import__(handler.required_package)
                except ImportError as err:
                    raise ImportError(
                        f"Loading {filename} requires {handler.required_package}. "
                        f"Install with: pip install {handler.required_package}"
                    ) from err
            return handler

    # No handler found
    supported = sorted({ext for h in FORMAT_HANDLERS for ext in h.extensions})
    raise ValueError(
        f"Cannot auto-detect format for '{filename}'. "
        f"Supported extensions: {', '.join(supported)}. "
        f"Use 'loader' parameter for custom formats."
    )


def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions.

    Returns:
        Sorted list of supported extensions
    """
    extensions = set()
    for handler in FORMAT_HANDLERS:
        extensions.update(handler.extensions)
    return sorted(extensions)
