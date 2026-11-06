# Agentic coding test

Does the model, with its drafter and a 4-bit KV cache, hold up as a coding agent? A real agent harness (opencode) drives
llama-server through eight programming tasks; hidden tests score the result and `analyze.py` looks for garbage in
everything the model wrote.

**Tasks** (`tasks/<name>/`: `repo/` the starting tree, `prompts/N.md` the turns, `hidden/` tests overlaid afterwards,
`ref/` a reference solution, `task.json` the hidden and suite commands):

| task | what |
|---|---|
| `py-orders-bug` | a support ticket: bulk discount charged twice across a small shop package |
| `py-notes-feature` | add a `search` command with filters to a CLI, document and test it |
| `py-refactor` | replace `get_user` with `fetch_user(..., include_deleted=)` across 7 modules, soft-deleted users |
| `mit-windowed` | a planted bug in more-itertools (5.5k-line module), from an issue report |
| `rust-lru` | an O(1) LRU cache from a stubbed API (std only) |
| `rust-expr` | tokenizer, recursive-descent parser and evaluator from a spec, with exact error positions |
| `js-csv` | fix a CSV parser until a failing test suite passes |
| `md-session` | four turns building a Markdown converter; hidden tests after every turn |

`python3 run.py X --validate` checks the harness itself: every reference passes its hidden tests and every untouched
repo fails them.

**Running:**
- `SERVER=1 ./build-llama.sh`, then `accuracy/agentic/serve.sh LOG` (`KV=q4|f16`, `NODRAFT=1`, `CTX`, `UB`) starts
  llama-server on port 8099.
- `AGENTIC_ENV=<toolchain> AGENTIC_SCRATCH=<dir> python3 run.py LABEL [--tasks a,b] [--reps N] --server-log LOG` runs
  the tasks. The toolchain is a nix `buildEnv` with python3 + pytest, cargo, rustc, node and coreutils.
- `campaign.sh SCRATCH REPS` runs the q4 cache, then the f16 cache, then the analysis.
- Each run is sandboxed by `sandbox.sh` (bubblewrap):
  - the filesystem is read-only, with `/home` and `/tmp` empty;
  - only the run's directory is writable;
  - the hidden tests run without network.
- Results go to `accuracy/results/agentic/LABEL/<task>-r<k>/`: the event streams, the exported session (with the
  reasoning), the final diff, the test output and the llama-server log slice.
- `python3 analyze.py LABEL...` writes `analysis.json` and prints the table.
