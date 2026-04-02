.PHONY: up down install ingest benchmark report test clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start postgres + pgvector via Docker Compose
	docker compose up -d
	@echo "Waiting for postgres..." && sleep 2
	python scripts/init_db.py

down: ## Stop and remove containers
	docker compose down

install: ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

ingest: ## Download and prepare SQuAD-200 benchmark corpus
	python scripts/ingest.py --corpus squad-200

benchmark: ## Run full benchmark: all strategies × all retrievers
	python scripts/run_benchmark.py --all-strategies --all-retrievers

report: ## Generate Markdown report from latest benchmark results
	python scripts/generate_report.py --latest

test: ## Run test suite
	pytest -q

clean: ## Remove results cache, judge cache, and reset database tables
	rm -f .cache/judge.json
	docker compose exec postgres psql -U lab -d raglab -c "TRUNCATE chunks, corpora;"
