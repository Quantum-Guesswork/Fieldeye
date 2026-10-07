.PHONY: install test dane demo demo-live gui clean

install:
	pip install -r requirements.txt pytest

test:
	python -m pytest tests/ -v

dane:
	python scripts/make_test_data.py dane/test --rows 3 --cols 6 --boustrophedon

demo: dane
	python -m stitchlive.cli batch dane/test/images -m detail --szwy -o wyniki

demo-live: dane
	@echo "Uruchom w drugim terminalu:"
	@echo "  ./scripts/simulate_flight.sh dane/test/images dane/naplyw 2"
	python -m stitchlive.cli watch dane/naplyw --okno 6 --timeout 20 -o wyniki

clean:
	rm -rf dane wyniki .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

gui:
	streamlit run app.py
