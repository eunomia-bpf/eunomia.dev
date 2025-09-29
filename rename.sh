# find ./tutorial -type f -name "*.md" ! -name "*_en*" -exec bash -c 'mv "$1" "${1%.md}.zh.md"' bash {} \; && \
# find ./tutorial -type f -name "*_en*.md" -exec bash -c 'mv "$1" "${1//_en/}"' bash {} \; && \
mkdir -p docs/setup && \
mkdir -p docs/tutorials && \
mkdir -p docs/others/cuda-tutorial && \
mkdir -p docs/others/cupti-tutorial && \
cp -rf tutorial/src/* docs/tutorials/ && \
mv tutorial/src/SUMMARY.zh.md docs/tutorials/index.zh.md && \
mv tutorial/src/SUMMARY.md docs/tutorials/index.md && \
mkdir -p docs/tutorials/imgs && \
cp tutorial/imgs/* docs/tutorials/imgs/ && \
cp -rf cuda-exp/* docs/others/cuda-tutorial/ && \
cp -rf cupti-exp/* docs/others/cupti-tutorial/ && \
cp bpftime/usage.md docs/bpftime/documents/usage.md && \
cp bpftime/installation.md docs/bpftime/documents/build-and-test.md && \
cp bpftime/example/README.md docs/bpftime/documents/examples.md  && \
cp bpftime/benchmark/README.md docs/bpftime/documents/performance.md && \
echo "eunomia.dev" > docs/CNAME
