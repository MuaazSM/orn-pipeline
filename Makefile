.PHONY: dev backend frontend pipeline install

# Run both backend and frontend concurrently
dev:
	@echo "Starting backend (port 8000) and frontend (port 3000)..."
	@trap 'kill 0' EXIT; \
	cd backend && ../venv/bin/uvicorn api.main:app --reload --port 8000 & \
	cd frontend && npm run dev & \
	wait

# Backend only
backend:
	cd backend && ../venv/bin/uvicorn api.main:app --reload --port 8000

# Frontend only
frontend:
	cd frontend && npm run dev

# Run ML pipeline
pipeline:
	cd backend && ../venv/bin/python run_pipeline.py --data ../../Final_final_used_data_Anonomyzed.csv

# Install all dependencies
install:
	python3 -m venv venv && venv/bin/pip install -r backend/requirements.txt
	cd frontend && npm install
