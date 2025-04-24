#!/bin/bash
set -e

python3 uni_ring_allreduce.py --npus_count=4 > uni_ring_allreduce.xml
