# find ./tutorial -type f -name "*.md" ! -name "*_en*" -exec bash -c 'mv "$1" "${1%.md}.zh.md"' bash {} \; && \
# find ./tutorial -type f -name "*_en*.md" -exec bash -c 'mv "$1" "${1//_en/}"' bash {} \; && \
mkdir -p docs/setup && \
mkdir -p docs/tutorials && \
mkdir -p docs/others/cuda-tutorial && \
mkdir -p docs/others/cupti-tutorial && \
mkdir -p docs/others/nvbit-tutorial && \
cp -rf tutorial/src/* docs/tutorials/ && \
mv tutorial/src/SUMMARY.zh.md docs/tutorials/index.zh.md && \
mv tutorial/src/SUMMARY.md docs/tutorials/index.md && \
mkdir -p docs/tutorials/imgs && \
cp tutorial/imgs/* docs/tutorials/imgs/ && \
cp -rf cuda-exp/* docs/others/cuda-tutorial/ && \
cp -rf cupti-exp/* docs/others/cupti-tutorial/ && \
cp -rf nvbit-tutorial/* docs/others/nvbit-tutorial/ && \
cp bpftime/usage.md docs/bpftime/documents/usage.md && \
cp bpftime/installation.md docs/bpftime/documents/build-and-test.md && \
cp bpftime/example/README.md docs/bpftime/documents/examples.md  && \
cp bpftime/benchmark/README.md docs/bpftime/documents/performance.md && \
cp bpftime/tools/bpftimetool/README.md docs/bpftime/documents/bpftimetool.md && \
cp bpftime/tools/aot/README.md docs/bpftime/documents/bpftimeaot.md && \
cp bpftime/attach/README.md docs/bpftime/documents/attach.md && \
cp bpftime/example/gpu/README.md docs/bpftime/documents/gpu.md && \
mkdir -p docs/agentsight && \
cp agentsight/README.md docs/agentsight/index.md && \
cp agentsight/README.zh-CN.md docs/agentsight/index.zh.md && \
cp agentsight/docs/usage.md docs/agentsight/quickstart.md && \
cp agentsight/docs/usage.zh-CN.md docs/agentsight/quickstart.zh.md && \
cp agentsight/docs/build.md docs/agentsight/build.md && \
cp agentsight/docs/docker.md docs/agentsight/docker.md && \
cp agentsight/docs/agents.md docs/agentsight/agents.md && \
cp agentsight/docs/otel.md docs/agentsight/otel.md && \
cp agentsight/docs/development.md docs/agentsight/development.md && \
cp agentsight/docs/development.zh-CN.md docs/agentsight/development.zh.md && \
mkdir -p docs/agentsight/images && \
cp agentsight/docs/demo-*.png agentsight/docs/top-mode-demo.png docs/agentsight/images/ && \
echo "eunomia.dev" > docs/CNAME
