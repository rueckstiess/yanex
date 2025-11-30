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

    name: str
    extensions: list[str]
    type_check: Callable[[Any], bool]
    saver: Callable[..., None]  # (obj, path, **kwargs) -> None
    loader: Callable[[Path], Any]
    required_package: str | None = None


def _save_text(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save string as text file.

    Supported kwargs:
        encoding: Text encoding (default: "utf-8")
    """
    if not isinstance(obj, str):
        raise TypeError(f"Expected str for .txt file, got {type(obj).__name__}")
    encoding = kwargs.get("encoding", "utf-8")
    path.write_text(obj, encoding=encoding)


def _load_text(path: Path) -> str:
    """Load text file as string."""
    return path.read_text(encoding="utf-8")


def _save_json(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save object as JSON.

    Supported kwargs:
        indent: Indentation level (default: 2)
        ensure_ascii: Escape non-ASCII characters (default: True)
        sort_keys: Sort dictionary keys (default: False)
    """
    # Extract JSON-specific kwargs with defaults
    indent = kwargs.get("indent", 2)
    ensure_ascii = kwargs.get("ensure_ascii", True)
    sort_keys = kwargs.get("sort_keys", False)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)


def _load_json(path: Path) -> Any:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_jsonl(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save list of objects as JSON Lines.

    Supported kwargs:
        ensure_ascii: Escape non-ASCII characters (default: True)
    """
    if not isinstance(obj, list):
        raise TypeError(f"Expected list for .jsonl file, got {type(obj).__name__}")
    ensure_ascii = kwargs.get("ensure_ascii", True)
    with path.open("w", encoding="utf-8") as f:
        for item in obj:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines file as list of dicts."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _save_csv_pandas(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save pandas DataFrame as CSV.

    Supported kwargs:
        index: Write row names (default: False)
        sep: Field separator (default: ",")
        All other kwargs are passed to DataFrame.to_csv()
    """
    import pandas as pd

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame for .csv file, got {type(obj).__name__}"
        )
    # Set default for index if not provided
    if "index" not in kwargs:
        kwargs["index"] = False
    obj.to_csv(path, **kwargs)


def _load_csv_pandas(path: Path) -> Any:
    """Load CSV file as pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(path)


def _save_csv_list(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save list of dicts as CSV.

    Supported kwargs:
        delimiter: Field delimiter (default: ",")
    """
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

    delimiter = kwargs.get("delimiter", ",")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=obj[0].keys(), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(obj)


def _save_npy(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save numpy array as .npy.

    Supported kwargs:
        allow_pickle: Allow saving object arrays using pickle (default: True)
    """
    import numpy as np

    if not isinstance(obj, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray for .npy file, got {type(obj).__name__}"
        )
    allow_pickle = kwargs.get("allow_pickle", True)
    np.save(path, obj, allow_pickle=allow_pickle)


def _load_npy(path: Path) -> Any:
    """Load .npy file as numpy array."""
    import numpy as np

    return np.load(path)


def _save_npz(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save dict of numpy arrays as .npz.

    Supported kwargs:
        compressed: Use compression (default: False)
    """
    import numpy as np

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict for .npz file, got {type(obj).__name__}")
    # Verify all values are numpy arrays
    for key, value in obj.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Expected dict of numpy arrays for .npz file, got dict with {type(value).__name__} at key '{key}'"
            )
    compressed = kwargs.get("compressed", False)
    if compressed:
        np.savez_compressed(path, **obj)
    else:
        np.savez(path, **obj)


def _load_npz(path: Path) -> dict[str, Any]:
    """Load .npz file as dict of numpy arrays."""
    import numpy as np

    loaded = np.load(path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def _save_torch(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save object with torch.save.

    Supported kwargs:
        pickle_protocol: Pickle protocol version (default: torch default)
        All other kwargs are passed to torch.save()
    """
    import torch

    torch.save(obj, path, **kwargs)


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


def _save_pickle(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save object with pickle.

    Supported kwargs:
        protocol: Pickle protocol version (default: pickle.HIGHEST_PROTOCOL)
    """
    protocol = kwargs.get("protocol", pickle.HIGHEST_PROTOCOL)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def _load_pickle(path: Path) -> Any:
    """Load object with pickle."""
    with path.open("rb") as f:
        return pickle.load(f)


def _save_matplotlib_figure(obj: Any, path: Path, **kwargs: Any) -> None:
    """Save matplotlib figure as image.

    Supported kwargs:
        dpi: Resolution in dots per inch (default: matplotlib's default)
        bbox_inches: Bounding box (default: None, use "tight" to trim whitespace)
        facecolor: Figure facecolor
        transparent: Make background transparent (default: False)
        All other kwargs are passed to Figure.savefig()
    """
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
    obj.savefig(path, **kwargs)


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
        name="text",
        extensions=[".txt"],
        type_check=lambda obj: isinstance(obj, str),
        saver=_save_text,
        loader=_load_text,
    ),
    # CSV with pandas
    FormatHandler(
        name="csv-pandas",
        extensions=[".csv"],
        type_check=lambda obj: type(obj).__name__ == "DataFrame",
        saver=_save_csv_pandas,
        loader=_load_csv_pandas,
        required_package="pandas",
    ),
    # CSV with list of dicts
    FormatHandler(
        name="csv-list",
        extensions=[".csv"],
        type_check=lambda obj: isinstance(obj, list)
        and (not obj or isinstance(obj[0], dict)),
        saver=_save_csv_list,
        loader=_load_csv_pandas,  # Load as pandas by default
        required_package="pandas",
    ),
    # JSON
    FormatHandler(
        name="json",
        extensions=[".json"],
        type_check=lambda obj: True,  # JSON handles many types
        saver=_save_json,
        loader=_load_json,
    ),
    # JSON Lines
    FormatHandler(
        name="jsonl",
        extensions=[".jsonl"],
        type_check=lambda obj: isinstance(obj, list),
        saver=_save_jsonl,
        loader=_load_jsonl,
    ),
    # NumPy array
    FormatHandler(
        name="npy",
        extensions=[".npy"],
        type_check=lambda obj: type(obj).__name__ == "ndarray",
        saver=_save_npy,
        loader=_load_npy,
        required_package="numpy",
    ),
    # NumPy arrays dict
    FormatHandler(
        name="npz",
        extensions=[".npz"],
        type_check=lambda obj: isinstance(obj, dict)
        and all(type(v).__name__ == "ndarray" for v in obj.values()),
        saver=_save_npz,
        loader=_load_npz,
        required_package="numpy",
    ),
    # PyTorch
    FormatHandler(
        name="torch",
        extensions=[".pt", ".pth"],
        type_check=lambda obj: True,  # torch.save handles any object
        saver=_save_torch,
        loader=_load_torch,
        required_package="torch",
    ),
    # Pickle (fallback for any object)
    FormatHandler(
        name="pickle",
        extensions=[".pkl"],
        type_check=lambda obj: True,  # Pickle handles any object
        saver=_save_pickle,
        loader=_load_pickle,
    ),
    # Matplotlib figure / PNG image
    FormatHandler(
        name="png",
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


def get_handler_for_load(filename: str, format: str | None = None) -> FormatHandler:
    """Find appropriate handler for loading based on filename extension or explicit format.

    Lookup order:
    1. If format specified, find by name
    2. Otherwise, find by extension (existing behavior)

    Args:
        filename: Filename to load (extension determines format if format not specified)
        format: Optional format name for explicit lookup

    Returns:
        FormatHandler for the filename

    Raises:
        ValueError: If format not found or no handler found for the extension
        ImportError: If required package is not installed
    """
    if format:
        # Explicit format - find by name
        for handler in FORMAT_HANDLERS:
            if handler.name == format:
                # Check if required package is available
                if handler.required_package:
                    try:
                        __import__(handler.required_package)
                    except ImportError as err:
                        raise ImportError(
                            f"Loading with format '{format}' requires {handler.required_package}. "
                            f"Install with: pip install {handler.required_package}"
                        ) from err
                return handler

        # Format not found
        available = sorted(h.name for h in FORMAT_HANDLERS)
        raise ValueError(
            f"Unknown format: '{format}'. Available formats: {', '.join(available)}"
        )

    # Auto-detect from extension
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


def register_format(
    name: str,
    extensions: list[str],
    type_check: Callable[[Any], bool],
    saver: Callable[[Any, Path], None],
    loader: Callable[[Path], Any],
    required_package: str | None = None,
) -> None:
    """Register a custom artifact format handler.

    Registered handlers are checked before built-in handlers during save operations,
    allowing custom types to override default behavior for specific extensions.

    Args:
        name: Format identifier for explicit loading (e.g., "workload")
        extensions: File extensions this handler supports (e.g., [".jsonl"])
        type_check: Function to check if object matches this handler
        saver: Function to save object to path: (obj, path) -> None
        loader: Function to load object from path: (path) -> object
        required_package: Optional package name required for this handler

    Raises:
        ValueError: If format name already registered or no extensions provided

    Examples:
        # Register workload format
        yanex.register_format(
            name="workload",
            extensions=[".jsonl"],
            type_check=lambda obj: isinstance(obj, Workload),
            saver=lambda obj, path: obj.save(str(path)),
            loader=lambda path: Workload.load(str(path)),
        )

        # Register format requiring optional dependency
        yanex.register_format(
            name="parquet",
            extensions=[".parquet"],
            type_check=lambda obj: isinstance(obj, pd.DataFrame),
            saver=lambda obj, path: obj.to_parquet(path),
            loader=lambda path: pd.read_parquet(path),
            required_package="pyarrow",
        )

        # Use registered handler
        yanex.save_artifact(workload, "data.jsonl")  # Auto-detects Workload type
        loaded = yanex.load_artifact("data.jsonl", format="workload")
    """
    # Validate name is unique
    if any(h.name == name for h in FORMAT_HANDLERS):
        raise ValueError(f"Format '{name}' is already registered")

    # Validate extensions
    if not extensions:
        raise ValueError("At least one extension must be provided")

    # Add to registry (prepend for priority over built-in handlers)
    handler = FormatHandler(
        name=name,
        extensions=extensions,
        type_check=type_check,
        saver=saver,
        loader=loader,
        required_package=required_package,
    )
    FORMAT_HANDLERS.insert(0, handler)
