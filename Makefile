.PHONY: build clean
build: docs/CNAME
	pip install mkdocs-material=="9.*" mkdocs-static-i18n=="0.53"
	mkdocs build

tutorial:
	git clone https://github.com/eunomia-bpf/bpf-developer-tutorial tutorial --depth=1

eunomia-bpf:
	git clone https://github.com/eunomia-bpf/eunomia-bpf eunomia-bpf --depth=1

GPTtrace:
	git clone https://github.com/eunomia-bpf/GPTtrace --depth=1

bpftime:
	git clone https://github.com/eunomia-bpf/bpftime --depth=1

docs/CNAME: tutorial eunomia-bpf GPTtrace bpftime
	./rename.sh

clean:
	rm -rf eunomia-bpf tutorial site GPTtrace docs/tutorials docs/setup docs/GPTtrace bpftime