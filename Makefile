.PHONY: build clean install

build: install
	mkdocs build

install: docs/CNAME
	pip install mkdocs-material=="9.*" mkdocs-static-i18n=="0.53"
	pip install "mkdocs-material[imaging]"
	pip install mkdocs-git-revision-date-localized-plugin
	pip install mkdocs-git-authors-plugin mkdocs-awesome-pages-plugin
	pip3 install mkdocs-exclude
	pip install mkdocs-rss-plugin

tutorial:
	git clone https://github.com/eunomia-bpf/bpf-developer-tutorial tutorial --depth=1

bpftime:
	git clone https://github.com/eunomia-bpf/bpftime --depth=1

docs/CNAME: tutorial
	./rename.sh

clean:
	rm -rf tutorial site docs/tutorials
