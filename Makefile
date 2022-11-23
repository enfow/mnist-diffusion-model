IMAGE_NAME=mnist-diffusion-model

format:
	black src/
	isort src/

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -v $(shell pwd):/tmp/ -it $(IMAGE_NAME) python $(FILE)
