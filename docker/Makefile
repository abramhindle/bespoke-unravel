.built: Dockerfile
	docker build . -t unravel
	touch .built

build: .built

shell: build
	docker run -it unravel /bin/bash
