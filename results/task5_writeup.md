# Task 5 — Alpha Documentation Contradiction

Ref: `phase0-closeout-handoff.md` Task 5.

## Fix

- `configs/default.yaml`: added a header comment stating this is the
  original Phase 1 config, never produced a reported result, and
  pointing readers to `configs/v2.yaml` onward as the actual source of
  truth for any reported number.
- `README.md`: added an "Alpha History" section with a table of every
  alpha value used across `default`/`v2`/`v3`/`v4`/`v5`, why each was
  chosen, and the explicit caveat text the handoff asked for (adapted to
  cover the full v2→v5 progression, not just the original v2=0.02 vs
  default=0.1 mismatch, since alpha has moved twice more since the
  handoff was written).

## Caveat text (as delivered)

> None of these alpha values should be read as "what a real deployed
> watermark would use" — all were chosen to make this research pipeline
> answer methodology questions (class balance, residue detectability),
> not to model production watermark strength.

## Verdict

Documented, not silently fixed — `default.yaml`'s alpha=0.1 is left
as-is (it's historically accurate for what Phase 1 used), with the
contradiction resolved by pointing to the real source of truth instead
of overwriting history.
