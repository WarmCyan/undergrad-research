all:
	python train.py --env-id SpaceInvaders-v0 --visualise --num-workers 4

clean:
	@rm -rfd tmp/*
	@rm -rfd runs/*
	@echo "" > out.txt
	@echo "" > error.txt

.PHONY: clean all
