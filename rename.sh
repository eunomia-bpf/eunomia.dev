find ./tutorial -type f -name "*.md" ! -name "*_en*" -exec bash -c 'mv "$1" "${1%.md}.zh.md"' bash {} \; && \
find ./tutorial -type f -name "*_en*.md" -exec bash -c 'mv "$1" "${1//_en/}"' bash {} \; && \
cp -rf eunomia-bpf/documents/src docs && \
mkdir -p docs/GPTtrace && \
cp GPTtrace/README.md docs/GPTtrace/index.md && \
cp -rf GPTtrace/doc docs/doc && \
mkdir -p docs/tutorials && \
cp -rf tutorial/src/* docs/tutorials && \
mv tutorial/README.zh.md docs/tutorials/SUMMARY.zh.md && \
mv tutorial/README.md docs/tutorials/SUMMARY.md && \
mv tutorial/src/SUMMARY.zh.md docs/tutorials/index.zh.md && \
mv tutorial/src/SUMMARY.md docs/tutorials/index.md && \
mv tutorial/imgs docs/tutorials/imgs && \
echo "eunomia.dev" > docs/CNAME