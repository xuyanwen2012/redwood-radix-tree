#!/bin/bash

cmake -S . -B build/ -G Ninja -D CMAKE_BUILD_TYPE:STRING=Release

cmake --build build/ --target all -- -j 4

