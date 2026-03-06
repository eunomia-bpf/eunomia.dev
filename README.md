# eunomia-bpf mainpage



This is the source of the official [eunomia-bpf website][official website].

For the tutorial, please edit https://github.com/eunomia-bpf/bpf-developer-tutorial

For the home page of each project, please edit the README of them.

[official website]: https://eunomia.dev

## Requirements

- [mkdocs](https://www.mkdocs.org/)
- [mkdocs-i18n](https://pypi.org/project/mkdocs-i18n/)
- [mkdocs-material](https://squidfunk.github.io/mkdocs-material/)
- Python 3.10+

## Local development

Clone this repo and enter, then:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdocs build
```

This will build `site` directory. Or:

```bash
mkdocs serve
```

The website will now be accessible at http://localhost:8000 and reload on any changes.
