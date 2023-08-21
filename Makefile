.PHONY: build clean
build: docs/CNAME
	pip install mkdocs-material=="9.*" mkdocs-static-i18n=="0.53"
	mkdocs build -v

tutorial:
	git clone https://github.com/eunomia-bpf/bpf-developer-tutorial tutorial

eunomia-bpf:
	git clone https://github.com/eunomia-bpf/eunomia-bpf eunomia-bpf

GPTtrace:
	git clone https://github.com/eunomia-bpf/GPTtrace

docs/CNAME: tutorial eunomia-bpf GPTtrace
	./rename.sh

clean:
	rm -rf eunomia-bpf tutorial docs site GPTtrace