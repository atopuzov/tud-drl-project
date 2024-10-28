.PHONY: help
help:
	@echo "No one can help us but ourselves"

.PHONY: tensorboard
tensorboard:
	@tensorboard --logdir logs

.PHONY: conda
conda:
	source /c/ProgramData/anaconda3/Scripts/activate tetris

.PHONY: fmt
fmt:
	python -m black .
	 python -m isort .

.CLEAN: clean
clean:
	rm -rf models/ best_model/ logs/
