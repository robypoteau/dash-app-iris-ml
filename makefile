all:
	$(error please pick a target)

env:
	# Create venv directory if not exist
	test -d env || virtualenv env
	./env/bin/python -m pip install -r requirements.txt

dev-env: env
	./env/bin/python -m pip install -r requirements-dev.txt

test:
	find . -name '*.pyc' -exec rm -f {} \;
	./env/bin/flake8 app tests
	./env/bin/python -m pytest \
	    --doctest-modules \
	    --disable-warnings \
	    --verbose \
	    app tests

run:
		export FLASK_ENV=development; \
		export FLASK_APP=main.py; \
		flask run

package:
	python setup.py sdist

clean:
	rm -rf build dist nlp.egg-info
