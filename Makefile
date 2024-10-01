.PHONY: build clean install

build: install
	mkdocs build -v

install: docs/CNAME
	pip install mkdocs-material=="9.*" mkdocs-static-i18n=="0.53"
	pip install "mkdocs-material[imaging]"
	pip install mkdocs-git-revision-date-localized-plugin
	pip install mkdocs-git-authors-plugin

tutorial:
	git clone https://github.com/eunomia-bpf/bpf-developer-tutorial tutorial --depth=1

eunomia-bpf:
	git clone https://github.com/eunomia-bpf/eunomia-bpf eunomia-bpf --depth=1

GPTtrace:
	git clone https://github.com/eunomia-bpf/GPTtrace --depth=1

bpftime:
	git clone https://github.com/eunomia-bpf/bpftime --depth=1

docs/CNAME: tutorial eunomia-bpf GPTtrace bpftime llvmbpf
	./rename.sh

llvmbpf:
	git clone https://github.com/eunomia-bpf/llvmbpf --depth=1

clean:
	rm -rf eunomia-bpf tutorial site GPTtrace docs/tutorials docs/eunomia-bpf/setup docs/GPTtrace bpftime llvmbpf