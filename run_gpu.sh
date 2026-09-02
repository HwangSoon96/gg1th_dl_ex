#!/bin/bash

export LD_LIBRARY_PATH=$(find .venv/lib/python*/site-packages/nvidia \
  -type d -name lib | tr '\n' ':'):$LD_LIBRARY_PATH

code .