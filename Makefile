PYTHON		:= python3
VENV_DIR	:= .venv
VENV_PY		:= $(VENV_DIR)/bin/python
VENV_PIP	:= $(VENV_DIR)/bin/pip

TRAINING_SCRIPT		:= training.py
PREDICT_SCRIPT		:= predict.py
PRECISION_SCRIPT	:= precision.py

MODEL_DIR		:= model
DATA_DIR		:= datasets

setup:
	@echo "Setting up virtual environment and installing dependencies..."
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_PIP) install --upgrade pip -q > /dev/null
	@$(VENV_PIP) install -r requirements.txt -q
	@echo "Setup complete."

clean:
	@rm -rf $(VENV_DIR) $(MODEL_DIR) $(DATA_DIR)/test.csv $(DATA_DIR)/train.csv __pycache__ *.pyc

re: clean all

.PHONY: setup clean re