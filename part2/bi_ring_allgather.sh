#!/bin/bash
set -e

python3 bi_ring_allgather.py --npus_count=4 --chunks_per_npu=2 > bi_ring_allgather.xml
