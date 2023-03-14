SOURCES = $(wildcard *.py) $(wildcard */*.py) $(wildcard */*/*.py)

train:
	poetry run train

run-mlflow-ui:
	poetry run mlflow ui &
	sleep 2
	xdg-open http://127.0.0.1:5000 &

black:
	poetry run nox -rs black

mypy:
	poetry run nox -rs mypy

tags:
	ctags -f tags -R --fields=+iaS --extra=+q $(SOURCES)

include_tags:
	ctags -f include_tags -R --languages=python --fields=+iaS --extra=+q \
		.venv/lib/python3.9/

sync_with_git:
	git fetch
	git reset origin/main --hard

clean:
	rm -rf tags include_tags __pycache__ */__pycache__ */*/__pycache__

.PHONY: clean include_tags tags

