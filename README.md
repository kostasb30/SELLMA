# SELLMA
Candidate Research Assignment — SE.LLMA

# Weight-Space Programming (WSP) for Swedish Language Models — SE.LLMA Investigation

This repository contains a reproducible, lightweight experimental investigation evaluating whether **Weight-Space Programming** (*Takehi et al., 2026*) can suppress unwanted English code-switching ("language drift") in open-weight language models without gradient-based fine-tuning.

## Repository Structure
- `src/wsp_swedish_experiment.py`: End-to-end reproducible experiment comparing an unchanged baseline, prompt control, targeted rank-1 `mlp.down_proj` weight edits, and norm-matched random weight perturbations.
- `results/metrics.json`: Quantitative evaluation scores and qualitative outputs generated upon execution.
- `RESEARCH_NOTE.md`: Concise 2-page research note detailing the hypothesis, methodology, findings, and strategic recommendation for SE.LLMA.
- `se_llma_weight_space_programming.pdf`: Full candidate report.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/wsp_swedish_experiment.py
