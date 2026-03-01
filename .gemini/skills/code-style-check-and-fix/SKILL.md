---
name: code-style-check-and-fix
description: Use this skill to review code using external tools and fix according to their output.
---

# Code style check and fix skill

## Workflow

### 1. Collect errors using `ruff`:

Run command (format and check code):

```
uv run ruff format & uv run ruff check --fix
```

### 2. Fix errors

For each file try to fix errors. Follow these instructions:

- Do not use `--unsafe-fixes` option
- Do not make changes that affect functionality
- Do not fix errors that require changes in more than 2 lines of code.

### 3. Generate a summary about errors that should be fixed manually
