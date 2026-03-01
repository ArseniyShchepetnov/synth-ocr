---
name: code-analyzer
description: Performs a deep analysis of the code to ensure consistency, quality design, adherence to style guides, and to identify potential improvements.
---

# Code analyzer skill

Use this skill to perform a thorough review of existing code or proposed changes.

## Workflow

### 1. Automated Validation
Start by running the project's automated quality tools to catch low-hanging fruit.

- **Linting & Formatting**: Run `uv run ruff check .` and `uv run ruff format --check .`.
- **Type Checking**: Run `uv run mypy .`.
- **Testing**: Run `uv run pytest`.

### 2. Consistency Analysis
Check if the code follows the project's established patterns and the rules in `GEMINI.md`.

- **Imports**: Ensure absolute imports are used (no relative imports).
- **Types**: Verify that Python 3.11+ type hints are used everywhere.
- **Naming**: Check that naming conventions are consistent across the module and project.
- **Comments**: Ensure there are no inline comments.
- **Docstrings**: Verify brief docstrings exist for all public symbols.

### 3. Design & Architecture Review
Evaluate the structural quality of the code.

- **Abstraction**: Are the abstractions appropriate? Is there logic that could be consolidated?
- **Responsibility**: Does each class/function have a single, clear responsibility?
- **Redundancy**: Identify any "just-in-case" logic or redundant code.
- **Complexity**: Identify areas with high cyclomatic complexity (many branches/statements).

### 4. Proposals for Improvements
Identify specific ways to make the code better.

- **Performance**: Are there any obvious bottlenecks or inefficient operations (e.g., in image processing or data loading)?
- **Readability**: Can any complex logic be simplified or better named?
- **Safety**: Are there potential edge cases or error conditions not handled?

### 5. Final Report
Generate a concise report summarizing:
- **Violations**: Any deviations from `GEMINI.md` or project standards.
- **Design Observations**: Feedback on the architecture.
- **Actionable Improvements**: A prioritized list of suggested changes.
