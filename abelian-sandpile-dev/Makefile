PYTHON ?= python3

.PHONY: evaluate quick random report demo clean

evaluate:
	$(PYTHON) run_judge.py

quick:
	$(PYTHON) run_judge.py --quick

random:
	$(PYTHON) run_judge.py --random-only

report:
	$(PYTHON) run_judge.py --json-output solution/benchmark.json

demo:
	$(PYTHON) demo.py

clean:
	rm -rf build
