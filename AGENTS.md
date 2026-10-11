# Repository Checkpoint Policy

This repository is under active, iterative modification. Preserve a clean Git
checkpoint before every code change so any unsatisfactory result can be
reverted precisely.

## Before editing

1. Run `git status --short --branch`.
2. If the working tree is dirty, commit the current intended state first with a
   descriptive message. Do not begin a new modification on top of unexplained
   uncommitted changes unless the user explicitly asks to combine them.
3. Create an annotated checkpoint tag for the pre-change state:

   ```text
   checkpoint-YYYYMMDD-HHMM-<short-description>
   ```

4. Record in the user-facing response the pre-change commit hash and tag.

## After editing

1. Run the smallest relevant test set, and the full `unittest discover -s tests`
   suite when the change can affect shared behavior.
2. Commit the completed change with a message that states what changed.
3. Report the post-change commit hash and the pre-change checkpoint tag.
4. Do not push unless the user explicitly asks.

## Rollback

Prefer non-destructive recovery:

- restore selected files from a checkpoint:
  `git restore --source=<tag> -- <path>...`
- inspect differences:
  `git diff <tag> -- <path>...`
- create a rollback branch from a checkpoint:
  `git switch -c rollback-<description> <tag>`

Do not run `git reset --hard`, delete tags, or rewrite shared history unless the
user explicitly requests it.
