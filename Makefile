.PHONY: install run-openai grade clean

install:
	pip install -e .

run-openai:
	bench-run --provider openai

grade:
	bench-grade

clean:
	rm -rf __pycache__ src/multimodal_bench/__pycache__ .venv build dist *.egg-info
