#!/bin/bash
set -e

python3 bi_ring_allreduce.py --npus_count=4 --chunks_per_npu=2 > bi_ring_allreduce.xml
