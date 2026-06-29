from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from loguru import logger

from openai_compatible_server.backends.base import BaseModelBackend
from openai_compatible_server.backends.demo import DemoModelBackend
from openai_compatible_server.config import Settings


def create_model_backend(settings: Settings) -> BaseModelBackend:
    class_path = settings.model_backend_class
    if not class_path:
        backend_class: type[BaseModelBackend] = DemoModelBackend
    else:
        module_name, separator, class_name = class_path.rpartition(":")
        if not separator or not module_name or not class_name:
            raise ValueError(
                "MODEL_BACKEND_CLASS must use 'package.module:ClassName' "
                "or '/path/to/backend.py:ClassName'"
            )
        module = _load_backend_module(module_name)
        if not hasattr(module, class_name):
            raise AttributeError(f"Backend class {class_name!r} was not found in {module_name!r}")
        backend_class = getattr(module, class_name)
        if not isinstance(backend_class, type) or not issubclass(backend_class, BaseModelBackend):
            raise TypeError(f"{class_path} is not a BaseModelBackend subclass")

    backend = backend_class(
        settings.model_id,
        max_concurrency=settings.model_max_concurrency,
        stream_chunk_size=settings.model_stream_chunk_size,
    )
    logger.info(
        "Using model backend | backend={} | model={}",
        type(backend).__name__,
        backend.model_id,
    )
    return backend


def _load_backend_module(module_or_path: str) -> ModuleType:
    if module_or_path.lower().endswith(".py"):
        path = Path(module_or_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model backend file was not found: {path}")

        module_name = f"_openai_compatible_server_custom_{path.stem}_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load model backend file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        backend_dir = str(path.parent)
        sys_path_inserted = backend_dir not in sys.path
        if sys_path_inserted:
            sys.path.insert(0, backend_dir)
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        finally:
            if sys_path_inserted:
                sys.path.remove(backend_dir)
        return module

    try:
        return importlib.import_module(module_or_path)
    except ModuleNotFoundError as exc:
        if exc.name == module_or_path or module_or_path.startswith(f"{exc.name}."):
            raise ModuleNotFoundError(
                f"Could not import model backend module {module_or_path!r}. "
                "Use an installed Python module such as "
                "'openai_compatible_server.backends.custom:CustomModelBackend', "
                "or a .py file path such as "
                "'.\\my_backend.py:MyModelBackend'."
            ) from exc
        raise
