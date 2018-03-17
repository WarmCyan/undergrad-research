all:
	python train.py --env-id PongDeterministic-v3 --visualise

clean:
	@rm -rfd tmp/*
	@rm -rfd runs/*
	@echo "" > out.txt
	@echo "" > error.txt

.PHONY: clean all
