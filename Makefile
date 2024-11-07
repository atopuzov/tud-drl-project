.PHONY: help
help:
	@echo "No one can help us but ourselves"

.PHONY: tensorboard
tensorboard:
	@tensorboard --logdir logs

.PHONY: conda
conda:
	@echo source /c/ProgramData/anaconda3/Scripts/activate tetris

.PHONY: fmt
fmt:
	python -m black .
	 python -m isort .

.PHONY: clean
clean:
	rm -rf models/ best_model/ logs/ tetris_model.zip

.PHONY: pyenv-remove
pyenv-remove:
	pyenv uninstall -f tud-project

.PHONY: pyenv-install
pyenv-install:
	pyenv virtualenv 3.11.6 tud-project
	pyenv local tud-project
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

.PHONY: mp4
mp4:
	ffmpeg -i images/tetris.gif -level 3.0 -profile:v baseline -vcodec libx264 -pix_fmt yuv420p -movflags +faststart tetris.mp4
