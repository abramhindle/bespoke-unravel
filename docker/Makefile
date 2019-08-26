.built: Dockerfile run.sh
	docker build . -t unravel
	touch .built

build: .built
DOCKERFLAGS=-v $(shell pwd)/input:/root/input 
DOCKERFLAGS+= -v $(shell pwd)/output:/root/ouput 
input:
	mkdir input
output:
	mkdir output
shell: build input output
	docker run $(DOCKERFLAGS) -it unravel /bin/bash
dshell: build input output
	docker run $(DOCKERFLAGS) -it unravel
