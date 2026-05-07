# CUAD Sample — Data Licenses & Attribution

The files in this directory are extracted from the **Contract Understanding
Atticus Dataset (CUAD) v1**, released by The Atticus Project under the
[Creative Commons Attribution 4.0 International license (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).

Per the license, use and redistribution (including commercial use) are
permitted provided attribution is preserved.

## Source

- **Source URL:** <https://github.com/TheAtticusProject/cuad>
- **Source artifact:** `CUADv1.json` inside `data.zip`
- **Source SHA-256:** `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae`
- **DOI:** <https://doi.org/10.5281/zenodo.4595826>

## Citation

> Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021).
> *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*.
> NeurIPS 2021 Datasets & Benchmarks Track. arXiv:2103.06268.

## What's here

A **5-contract × 5-clause** subset of CUAD v1, chosen for diverse document
types (sponsorship / reseller / outsourcing / hosting / distributor
agreements) in the 8–16 KB range with full golden-label coverage on the
five target clauses. See `MANIFEST.json` for the exact mapping back to
CUAD indices and SHA-256s.

Each `*.txt` file contains the verbatim `context` field from CUAD for the
named contract (no modifications). The accompanying
`cuad-extraction-golden.jsonl` carries one JSON object per contract with
CUAD's ground-truth spans for:

- Parties
- Agreement Date
- Governing Law
- Termination For Convenience
- Cap On Liability

## Modifications

No text modifications. The following transformations were applied:

1. Renamed files to lowercase-hyphenated slugs derived from the CUAD
   contract titles (for filesystem friendliness).
2. Grouped CUAD's SQuAD-format annotations by clause type and emitted one
   JSON object per contract in JSONL.
3. Excluded the 36 clause types not targeted by this sample.

The full 41-clause × 510-contract dataset is not redistributed here; fetch
it from the upstream source linked above for full evaluation runs.
