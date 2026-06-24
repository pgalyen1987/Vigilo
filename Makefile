.PHONY: install dev build up down logs train detect report release clean lint test demo span-up help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Local development ───────────────────────────────────────────────

install: ## Install vigilo in editable mode
	pip install -e .

dev: ## Install with dev extras (pytest, ruff)
	pip install -e ".[dev]"

lint: ## Run ruff linter
	ruff check vigilo/ arch/

test: ## Run test suite
	pytest tests/ -v

demo: ## Generate bundled demo data + checkpoint
	python scripts/generate_demo_data.py

span-up: ## Start passive SPAN capture (Zeek + Vigilo)
	docker compose --profile span up -d

span-down: ## Stop SPAN profile services
	docker compose --profile span down

# ── Docker ──────────────────────────────────────────────────────────

build: ## Build the Docker image
	docker compose build

up: ## Start the dashboard (docker)
	docker compose up -d vigilo

down: ## Stop all containers
	docker compose down

logs: ## Tail container logs
	docker compose logs -f vigilo

# ── Release ─────────────────────────────────────────────────────────

release: ## Build release package (image + customer files)
	bash scripts/build-release.sh $(VERSION)

# ── Workflows ───────────────────────────────────────────────────────

train: ## Train the model (pass ARGS="--logs ...")
	vigilo train $(ARGS)

detect: ## Run detection (pass ARGS="--log ... --ckpt ...")
	vigilo detect $(ARGS)

report: ## Generate HTML report (pass ARGS="--log ... --out ...")
	vigilo report $(ARGS)

# ── Cleanup ─────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info __pycache__ vigilo/__pycache__ arch/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
