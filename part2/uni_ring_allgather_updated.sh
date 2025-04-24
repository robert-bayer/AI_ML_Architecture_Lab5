#!/bin/bash
set -e

python3 uni_ring_allgather_updated.py --npus_count=4 --chunks_per_npu=2 > uni_ring_allgather_updated.xml
