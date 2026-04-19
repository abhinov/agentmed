.PHONY: install run-openai grade clean prepare-batch retrieve-batch

install:
	pip3 install -e .

run-openai:
	bench-run --provider openai

grade:
	bench-grade

prepare-batch:
	python3 src/multimodal_bench/batch_openai.py prepare

retrieve-batch:
	python3 src/multimodal_bench/batch_openai.py retrieve --batch-id $(BATCH_ID)

clean:
	rm -rf __pycache__ src/multimodal_bench/__pycache__ .venv build dist *.egg-info
