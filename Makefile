.PHONY: *

install:
	python3.10 -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements.txt

train:
	venv/bin/python src/train.py

