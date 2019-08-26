#!/bin/bash
pushd /root/setupenv > /dev/null
source ./setupenv `./latest`
popd > /dev/null
exec $*
