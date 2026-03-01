"""Settings for transform discovery."""

import os

# List of modules to inspect for BaseTransform subclasses
TRANSFORM_MODULES = [
    ".background",
    ".distortion",
    ".misc",
    ".mix",
]

# Allow extending via environment variable (comma-separated list)
_extra_modules = os.getenv("OCR_EXTRA_TRANSFORMS", "")
if _extra_modules:
    for raw_mod in _extra_modules.split(","):
        mod = raw_mod.strip()
        if mod and mod not in TRANSFORM_MODULES:
            TRANSFORM_MODULES.append(mod)
