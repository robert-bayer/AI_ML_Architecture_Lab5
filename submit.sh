#!/bin/bash
set -e

# archive the files
zip -j LastName_FirstName_GTID_ECE_8803_HML_sp25_lab5.zip \
    part1/discussion.md \
    part2/uni_ring_allgather_updated.py \
    part2/bi_ring_allgather.py \
    part3/uni_ring_allreduce_updated.py \
    part3/bi_ring_allreduce.py \
    part4/uni_ring_mesh.py \
    part4/bi_ring_mesh.py \
    part5/hierarchical_mesh.py
