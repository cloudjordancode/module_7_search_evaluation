# Evaluate Your Search System

**Module:** 7 — Applied AI: Embeddings & Retrieval
**Estimated time:** 35 minutes

## Objective

Create an evaluation set for a semantic search system and use it to measure and improve search quality using precision and recall.

## What You'll Build

A script that sets up a 12+ document ChromaDB collection, defines 6+ test queries with expected relevant IDs, implements an `evaluate()` function that computes precision and recall, and runs the evaluation at 3 different settings (varying threshold or `n_results`).

## Reference Code

The starter file (`starter.py`) provides a scaffold with TODOs — fill in each section, then compare with the solution.

## Running

```bash
python starter.py
```

## Deliverable

A working evaluation script with 6+ test queries, precision/recall calculations at multiple settings, and a written analysis of which queries worked well and which failed.
