# Convenience commands. On Windows use Git Bash, WSL, or run the python
# commands directly (PowerShell does not run Makefiles natively).

install:
	pip install -r requirements.txt

init-db:
	python -m aqi_predictor.database.indexes

feature:
	python pipelines/feature_pipeline.py

backfill:
	python pipelines/backfill_pipeline.py --days 30

train:
	python pipelines/train_pipeline.py

predict:
	python pipelines/batch_predict_pipeline.py

api:
	uvicorn app.api.main:app --reload

dashboard:
	streamlit run app/dashboard/streamlit_app.py

test:
	pytest -q
