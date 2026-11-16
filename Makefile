# InferLab - common tasks.  `make help` lists them.
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PORT ?= 8000

.DEFAULT_GOAL := help
.PHONY: help venv install kernels kernels-test kernels-bench download serve dev test lint security audit \
        check labs docker docker-run secret clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## create a virtualenv in .venv
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

install: venv  ## install pinned dependencies + the package (editable)
	$(BIN)/pip install -q -r requirements-dev.txt
	$(BIN)/pip install -q -e . --no-deps

kernels:  ## build the C++ kernels (CMake, -O3 -march=native, hardened)
	cmake -S csrc -B csrc/build -DCMAKE_BUILD_TYPE=Release
	cmake --build csrc/build -j
	./csrc/build/test_kernels

kernels-test:  ## C++ kernel tests under AddressSanitizer + UBSan
	cmake -S csrc -B csrc/build-asan -DCMAKE_BUILD_TYPE=Debug -DINFERLAB_SANITIZE=ON
	cmake --build csrc/build-asan -j
	ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./csrc/build-asan/test_kernels

kernels-bench: kernels  ## standalone C++ benchmark
	./csrc/build/bench_kernels 1024

download:  ## download TinyStories Llama-2 15M + the Llama-2 tokenizer into ./models
	$(BIN)/inferlab download tokenizer.bin stories15M.bin

serve:  ## run the server + UI (auth on; a key is generated if none is set)
	$(BIN)/inferlab serve --port $(PORT)

dev:  ## run on 127.0.0.1 without auth (local development only)
	$(BIN)/inferlab serve --port $(PORT) --no-auth --kernels cpp

test:  ## run the Python test suite
	$(BIN)/pytest

lint:  ## ruff lint
	$(BIN)/ruff check inferlab tests labs benchmarks scripts

security:  ## static security analysis (bandit) + dependency audit
	$(BIN)/bandit -q -r inferlab labs benchmarks scripts -c pyproject.toml
	$(BIN)/pip-audit -r requirements.txt --strict

audit:  ## dependency vulnerability audit only
	$(BIN)/pip-audit -r requirements.txt --strict

check: lint test security kernels-test  ## everything CI runs

labs:  ## run every lab script in order
	@for f in labs/[0-9]*.py; do echo "\n======== $$f"; $(BIN)/python $$f || exit 1; done

secret:  ## create secrets/api_keys.txt for docker compose (dir 700, git-ignored)
	@mkdir -p secrets && chmod 700 secrets
	@test -s secrets/api_keys.txt || $(PY) -c "import secrets; print('il-' + secrets.token_urlsafe(32))" > secrets/api_keys.txt
	@chmod 644 secrets/api_keys.txt  # readable by the container's non-root user; the 700 dir protects it on the host
	@echo "API key is in secrets/api_keys.txt (git-ignored)"

docker:  ## build the container image
	docker build -t inferlab:1.0.0 .

docker-run: secret  ## run the hardened container (compose)
	docker compose up --build

clean:  ## remove build artefacts
	$(RM) -r csrc/build csrc/build-asan inferlab/kernels/lib .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec $(RM) -r {} +
