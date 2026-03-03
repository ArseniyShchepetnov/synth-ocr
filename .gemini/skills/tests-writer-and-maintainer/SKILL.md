---
name: tests-writer-and-maintainer
description: Use this skill to generate or review high-quality tests for the project.
---

## General

- Write tests in the `tests` folder
- Repeat directory structure similar to the source code
- If there are unclear or ambiguous information or decision ask user using `ask_user`.

## Code style guidelines

- Write clear and consistent tests in pytest framework
- Write brief description of test in docstring
- Use `tempfile` for generated file resources
- Fix seeds for probabilistic functions
- Use mocking only for external resources like: APIs, databases, etc.
