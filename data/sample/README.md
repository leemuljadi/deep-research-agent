# Deep-Research Agent — sample corpus

This folder is a tiny stand-in for real research source material. Drop any
`.txt`, `.md` or `.py` files in here (or another directory) and index them:

```bash
python -m scripts.ingest_corpus data/sample
```

## Why a sample corpus?

The agent grounds its answers in *retrieved* passages. Without a corpus there's
nothing to retrieve, so this file itself (and the project README) becomes the
first searchable material. Replace/extend it with real domain content for your
golden set.

## Example question to run against this corpus

```bash
python -m scripts.run_research "What does the Deep-Research Agent do and how is it evaluated?"
```
