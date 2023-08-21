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

tutorial/README.zh.md: tutorial eunomia-bpf GPTtrace
	./rename.sh

docs/CNAME: tutorial/README.zh.md
	cp -rf eunomia-bpf/documents/src docs
	mkdir -p docs/GPTtrace
	cp GPTtrace/README.md docs/GPTtrace/index.md
	mkdir -p docs/tutorials
	cp -rf tutorial/src/* docs/tutorials
	mv tutorial/README.zh.md docs/tutorials/SUMMARY.zh.md
	mv tutorial/README.md docs/tutorials/SUMMARY.md
	mv tutorial/src/SUMMARY.zh.md docs/tutorials/index.zh.md
	mv tutorial/src/SUMMARY.md docs/tutorials/index.md
	mv tutorial/imgs docs/tutorials/imgs
	echo "eunomia.dev" > docs/CNAME

clean:
	rm -rf eunomia-bpf tutorial docs site GPTtrace