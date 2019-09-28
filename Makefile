.PHONY: clean build run

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean          to remove build files"
	@echo "  build          to download and process data from raw"
	@echo "  train          to train model with default config"
	@echo "  freeze         to freeze model for serving"
	@echo "  predict        to extract keyphrase from sample text"

clean:
	rm -rf data/processed/*
	rm -rf data/external/*
	rm -rf data/interim/*
	rm -rf data/raw/*
	rm -rf models/*
	rm -rf results/*
	rm -rf visualization/*

build:
	python3 build_data.py

run:
	python3 run_k_mean.py
