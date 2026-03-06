.PHONY: build clean install

VENV := .venv
PYTHON := $(VENV)/bin/python
MKDOCS := $(VENV)/bin/mkdocs

build: install
	$(MKDOCS) build

install: $(VENV)/bin/activate docs/CNAME requirements.txt
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

tutorial:
	git clone https://github.com/eunomia-bpf/bpf-developer-tutorial tutorial --depth=1

bpftime:
	git clone https://github.com/eunomia-bpf/bpftime --depth=1

cuda-exp:
	git clone https://github.com/eunomia-bpf/basic-cuda-tutorial cuda-exp --depth=1

cupti-exp:
	git clone https://github.com/eunomia-bpf/cupti-tutorial cupti-exp --depth=1

nvbit-tutorial:
	git clone https://github.com/eunomia-bpf/nvbit-tutorial --depth=1

docs/CNAME: tutorial cuda-exp cupti-exp nvbit-tutorial bpftime
	./rename.sh

clean:
	rm -rf tutorial site docs/tutorials bpftime cuda-exp cupti-exp nvbit-tutorial
