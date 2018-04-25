all:
	#python train.py --env-id SpaceInvaders-v0 --visualise --num-workers 2
	#python train.py --env-id SpaceInvaders-v0 --visualise --num-workers 8
	#python train.py --env-id SpaceInvaders-v0 --num-workers 8
	#python train.py --env-id SpaceInvaders-v0 --num-workers 2
	#python train.py --env-id SpaceInvaders-v0 --num-workers 5 --visualise
	python train.py --env-id SpaceInvaders-v0 --num-workers 8 --visualise

clean:
	@rm -rfd tmp/*
	@rm -rfd runs/*
	@echo "" > out.txt
	@echo "" > error.txt

.PHONY: clean all
