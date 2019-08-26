.built: Dockerfile run.sh
	docker build . -t unravel
	touch .built

build: .built

shell: build
	docker run -it unravel /bin/bash
dshell: build
	docker run -it unravel
