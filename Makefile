# Makefile

# Define Python executable
PYTHON = python3

# Define the script name
SCRIPT = main.py

DATA_PATH = "data/OCR_Dates_Dataset/OCR_Dates/"

run:
	rm -rf model_deployment/media/temp/*
	python model_deployment/manage.py runserver

# Default target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make show_data      Show raw data images"
	@echo "  make train          Train the model"


# Show data images
.PHONY: show_data
show_data:
	$(PYTHON) $(SCRIPT) --show-some-data --data-path $(DATA_PATH)

# Train model
.PHONY: train
train:
	$(PYTHON) $(SCRIPT) --train

