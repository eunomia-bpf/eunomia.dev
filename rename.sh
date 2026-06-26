# find ./tutorial -type f -name "*.md" ! -name "*_en*" -exec bash -c 'mv "$1" "${1%.md}.zh.md"' bash {} \; && \
# find ./tutorial -type f -name "*_en*.md" -exec bash -c 'mv "$1" "${1//_en/}"' bash {} \; && \
mkdir -p docs/setup && \
mkdir -p docs/tutorials && \
mkdir -p docs/others/cuda-tutorial && \
mkdir -p docs/others/cupti-tutorial && \
mkdir -p docs/others/nvbit-tutorial && \
cp -rf tutorial/src/* docs/tutorials/ && \
if [ -f tutorial/src/SUMMARY.zh.md ]; then cp tutorial/src/SUMMARY.zh.md docs/tutorials/index.zh.md; fi && \
if [ -f tutorial/src/SUMMARY.md ]; then cp tutorial/src/SUMMARY.md docs/tutorials/index.md; fi && \
rm -f docs/tutorials/SUMMARY.md docs/tutorials/SUMMARY.zh.md && \
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
rm -rf docs/agentsight && \
mkdir -p docs/agentsight && \
rsync -a --exclude '/design/' --exclude '/experiment/' agentsight/docs/ docs/agentsight/ && \
cp agentsight/README.md docs/agentsight/index.md && \
cp agentsight/README.zh-CN.md docs/agentsight/index.zh.md && \
mkdir -p docs/agentsight/images && \
cp agentsight/docs/demo-*.png agentsight/docs/top-mode-demo.png docs/agentsight/images/ && \
mkdir -p docs/actplane && \
rm -rf docs/actplane && \
mkdir -p docs/actplane && \
rsync -a --exclude '/design/' --exclude '/papers/' actplane/docs/ docs/actplane/ && \
cp actplane/README.md docs/actplane/index.md && \
perl -0pi -e 's#\]\(docs/rule-language\.md\)#](rule-language.md)#g; s#\]\(script/CLAUDE\.snippet\.md\)#](https://github.com/eunomia-bpf/ActPlane/blob/master/script/CLAUDE.snippet.md)#g; s#\]\(bpf/\)#](https://github.com/eunomia-bpf/ActPlane/tree/master/bpf/)#g; s#\]\(LICENSE\)#](https://github.com/eunomia-bpf/ActPlane/blob/master/LICENSE)#g' docs/actplane/index.md && \
perl -0pi -e 's#\]\(design/feedback-design\.md\)#](https://github.com/eunomia-bpf/ActPlane/blob/master/docs/design/feedback-design.md)#g; s#\]\(\.\./script/agent-feedback\.md\)#](https://github.com/eunomia-bpf/ActPlane/blob/master/script/agent-feedback.md)#g' docs/actplane/rule-language.md && \
echo "eunomia.dev" > docs/CNAME
