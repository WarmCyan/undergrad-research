all: a3c

a3c:
	python A3C_rnn.py

tutorial:
	python A3C_actual.py

clean:
	@rm -rfd ../tensorboard_data/a3c_*
	@rm -rfd runs/*


.PHONY: all a3c clean tutorial
