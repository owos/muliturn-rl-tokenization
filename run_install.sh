#!/bin/bash

#This part is need for OSC users
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS2836/${USER}/triton_cache


export UV_CACHE_DIR=/fs/scratch/PAS2836/${USER}/.cache/uv  #control your uv caches


cd lm-evaluation-harness
uv pip install -e .
uv pip install "lm_eval[hf,vllm,api]"