.PHONY: clean deps build upload release brew-tap brew-formula brew-release

clean:
	rm -rf dist/ build/ *.egg-info

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

release: upload

deps:
	python -m pip install --upgrade build twine

man:
	pandoc -s -f markdown -t man README.md -V title="dataset-translator" -V section=1 -o docs/dataset-translator.1
