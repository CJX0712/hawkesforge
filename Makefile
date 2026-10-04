# HawkesForge Makefile — 一键复现（作者 晨星）
# 使用隔离 venv；若已激活自有环境，直接 make test / make demo 亦可。

PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
RUN := $(VENV)/bin/python

.PHONY: all venv install lint test demo ci clean

all: venv install lint test demo

venv:
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	@# Windows 下 bin -> Scripts，做一次兼容软链便于统一调用
	@test -e $(VENV)/bin || (cd $(VENV) && ln -sf Scripts bin)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install ruff==0.16.10 pytest pytest-cov

lint: venv
	$(RUN) -m ruff check .
	$(RUN) -m ruff format --check .

test: venv
	$(RUN) -m pytest -q

demo: venv
	$(RUN) examples/run_demo.py

ci: lint test demo

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .coverage benchmark.json demo_out.log
	find . -type d -name __pycache__ -exec rm -rf {} +
