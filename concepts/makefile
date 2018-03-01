all: a3c

a3c:
	python A3C.py

tutorial:
	python a3c_cartpole.py

clean:
	@rm -rfd ../tensorboard_data/a3c_*
	@rm -rfd runs/*


.PHONY: all a3c clean tutorial
