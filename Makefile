.PHONY: doctor train bench serve test

doctor:
	python -m beamforge doctor

train:
	python -m beamforge train

bench:
	python -m beamforge bench

serve:
	python -m beamforge serve

test:
	python -m unittest discover -s tests -t .
