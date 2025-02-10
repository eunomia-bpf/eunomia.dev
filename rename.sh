# find ./tutorial -type f -name "*.md" ! -name "*_en*" -exec bash -c 'mv "$1" "${1%.md}.zh.md"' bash {} \; && \
# find ./tutorial -type f -name "*_en*.md" -exec bash -c 'mv "$1" "${1//_en/}"' bash {} \; && \
mkdir -p docs/setup && \
mkdir -p docs/tutorials && \
cp -rf tutorial/src/* docs/tutorials/ && \
mv tutorial/src/SUMMARY.zh.md docs/tutorials/index.zh.md && \
mv tutorial/src/SUMMARY.md docs/tutorials/index.md && \
mkdir -p docs/tutorials/imgs && \
cp tutorial/imgs/* docs/tutorials/imgs/ && \
echo "eunomia.dev" > docs/CNAME