# Phantom or Real?

Investigating whether SAE features from Gemma Scope 2 predict Gemma 3 reasoning
failures on depth-controlled ontology tasks, and whether those features are
causally relevant or merely correlational "phantoms". See
`BEHAVIORAL_DATA_PLAN.md` for the full stage-1 spec and `CLAUDE.md` for the
operating layout of this repo.

## Quick start

```bash
# One-time setup
conda env create -f environment.yml            # or environment.lock.yml for exact pins
conda activate phantom

# You need a clone of beyond-deduction somewhere — options:
#   - set BD_PATH=/path/to/beyond-deduction
#   - or ln -s /path/to/beyond-deduction third_party_beyond_deduction
#   - or clone it to ~/beyond-deduction

# Generate the full dataset (~44k examples)
python -m src.generate_examples --counts full

# Pilot run (50 per height per task per model = 400 examples per model)
python -m src.generate_examples --counts pilot
# Then: see CLAUDE.md for inference commands
```

Credentials required for running inference / error classification are listed
in `CLAUDE.md`.

Thank you.
