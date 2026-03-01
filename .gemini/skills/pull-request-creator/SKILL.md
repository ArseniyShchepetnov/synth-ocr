---
name: pull-request-creator
description: Prepares a pull request by validating the code and generating a descriptive title and body using the project's PR template.
---

# Pull request creator skill

Use this skill to wrap up your changes and prepare them for submission as a pull request.

## Workflow

### 1. Pre-validation
Ensure that the code adheres to the project standards and is ready for submission.

- **Check Branch**: Ensure you are not on the `main` or `master` branch. If you are, propose creating a new branch.
- **Run Tests**: Run `uv run pytest` to ensure all tests pass.

### 2. Analyze Changes
Gather information about the changes to provide a high-quality description.

- **Git Status**: List all changed files.
- **Git Diff**: Review the differences compared to the base branch (usually `main`).
- **Git Log**: Review the commit history on the current branch.

### 3. Generate PR Description
Construct a high-quality pull request title and body following the structure in `.github/PULL_REQUEST_TEMPLATE.md`:

- **Title**: Use conventional commits format (e.g., `feat: add receipt data source`).
- **Body**: Follow the template strictly (Summary, Changes, Related Issues, Additional Context).
- **Temporary File**: Write the generated body to a temporary file named `PR_DESCRIPTION.md`.

### 4. Final Output
Present the prepared PR title and the explicit `gh` command to the user:

```bash
gh pr create --title "<PR_TITLE>" --body-file PR_DESCRIPTION.md
```

Inform the user that the PR body has been written to `PR_DESCRIPTION.md` and should be deleted after the PR is successfully created.
