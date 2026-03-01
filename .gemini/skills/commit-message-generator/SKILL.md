---
name: commit-message-generator
description: Generates a high-quality commit message in conventional commit format based on an analysis of changes.
---

# Commit message generator skill

Use this skill to generate a professional commit message that follows the conventional commits specification.

## Workflow

### 1. Analyze Changes
Gather context about the work performed to ensure the message is accurate and descriptive.

- **Git Status**: Verify which files are staged for commit.
- **Git Diff**: Review the exact changes in the staged files (using `--staged`).
- **Git Log**: Check recent commits to match the project's style and verbosity.

### 2. Generate Commit Message
Construct the message according to these standards:

- **Format**: `<type>[optional scope]: <description>`
- **Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- **Subject Line**:
    - Use the imperative, present tense ("change" not "changed").
    - Do not capitalize the first letter.
    - Do not end with a period.
- **Body** (if needed):
    - Provide a clear motivation for the change.
    - Explain "why" rather than "what" (the diff shows "what").
    - Use bullet points for multiple changes.

### 3. Final Output
Present the generated message to the user.
