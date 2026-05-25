## Quick start

```bash
git clone https://github.com/VChahar1/pairs-trading.git
cd pairs-trading
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v                                # ~18 tests
jupyter notebook notebooks/              # browse the analysis notebooks
```

The notebooks should be run in order (01 → 04). Intermediate artifacts are cached to `data/processed/` so later notebooks don't redo earlier computation.
