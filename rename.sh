find ./tutorial -type f -name "*.md" ! -name "*_en*" -exec bash -c 'mv "$1" "${1%.md}.zh.md"' bash {} \;
find ./tutorial -type f -name "*_en*.md" -exec bash -c 'mv "$1" "${1//_en/}"' bash {} \;
