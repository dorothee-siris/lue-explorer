# LUE Portfolio Explorer

A Streamlit app to explore research outputs with multiple dashboards. v0.1 focuses on the **Lab View** with side‑by‑side comparisons.

## Quick start


```bash
# 1) clone
git clone <your-fork-or-repo-url>
cd app


# 2) (optional) create venv
python -m venv .venv && source .venv/bin/activate


# 3) install deps
pip install -r requirements.txt


# 4) add data
mkdir -p data
# copy parquet files here:
# pubs_final.parquet
# dict_internal.parquet


# 5) run
streamlit run streamlit_app.py