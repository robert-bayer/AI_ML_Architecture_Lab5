#!/bin/bash
set -e

python3 uni_ring_allgather.py --npus_count=4 > uni_ring_allgather.xml
