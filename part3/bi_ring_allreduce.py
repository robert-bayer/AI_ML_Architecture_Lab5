import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce


def bi_ring_allreduce(npus_count: int, chunks_per_npu: int) -> None:
    topology = fully_connected(npus_count)
    
    # Note: now each NPU starts with:
    # C * N (=chunks_per_npu * npus_count) number of chunks
    # and ends with C * N number of chunks.
    # Assumption: C is even, so that each ring can process C/2 number of chunks.
    collective = AllReduce(num_ranks=npus_count, chunk_factor=chunks_per_npu * npus_count, inplace=True)

    with MSCCLProgram("bi_ring_allreduce", topology, collective, 1):
        ### ===============================================
        # Lab 5.3.2
        # TODO: Implement bidirectional Ring All-Reduce algorithm
        
        # Hint: modify your implementation of uni_ring_allreduce_updated.py
        
        # Hint: C/2 chunks follow the original Ring
        # Hint: and the other C/2 chunks follow the Ring in the opposite direction
        ### ===============================================
        for npu in range(npus_count):

            half_chunks = chunks_per_npu // 2

            chunk_start = npu * chunks_per_npu

            for ch in range(half_chunks):

                # this NPU will start sending out chunk (=index) npu.
                chunk_id = chunk_start + ch
                c = chunk(rank=npu, buffer=Buffer.input, index=chunk_id)
                
                # run Ring Reduce-Scatter
                # each chunk perspective, this process is:
                #   1. Send this chunk to the next NPU (npu + 1)
                #   2. The receiver NPU reduces the chunk with its own chunk
                #   3. Repeat this for (N-1) steps
                next = (npu + 1) % npus_count
                for step in range(npus_count - 1):
                    # corresponding chunk on the next NPU
                    next_c = chunk(rank=next, buffer=Buffer.input, index=chunk_id)
                    
                    # reduce to this chunk by sending c
                    c = next_c.reduce(c)
                    
                    # Note: now c denotes the reduced chunk on the next NPU (next),
                    # not the original chunk on the sender NPU (npu).
                    
                    # update next NPU
                    next = (next + 1) % npus_count

                # likewise, run Ring All-Gather.
                # currently, the final reduced chunk is at NPU (next - 1).
                # send the final reduced chunk to the next NPU
                # which takes N-1 steps
                for step in range(npus_count - 1):
                    # send the chunk to the next NPU
                    c = c.copy(dst=next, buffer=Buffer.input, index=chunk_id)
                    
                    # update next NPU
                    next = (next + 1) % npus_count
            for ch in range(half_chunks, chunks_per_npu):
                # this NPU will start sending out chunk (=index) npu.
                chunk_id = chunk_start + ch
                c = chunk(rank=npu, buffer=Buffer.input, index=chunk_id)
                
                # run Ring Reduce-Scatter
                # each chunk perspective, this process is:
                #   1. Send this chunk to the next NPU (npu + 1)
                #   2. The receiver NPU reduces the chunk with its own chunk
                #   3. Repeat this for (N-1) steps
                next = (npu - 1) % npus_count
                for step in range(npus_count - 1):
                    # corresponding chunk on the next NPU
                    next_c = chunk(rank=next, buffer=Buffer.input, index=chunk_id)
                    
                    # reduce to this chunk by sending c
                    c = next_c.reduce(c)
                    
                    # Note: now c denotes the reduced chunk on the next NPU (next),
                    # not the original chunk on the sender NPU (npu).
                    
                    # update next NPU
                    next = (next - 1) % npus_count

                # likewise, run Ring All-Gather.
                # currently, the final reduced chunk is at NPU (next - 1).
                # send the final reduced chunk to the next NPU
                # which takes N-1 steps
                for step in range(npus_count - 1):
                    # send the chunk to the next NPU
                    c = c.copy(dst=next, buffer=Buffer.input, index=chunk_id)
                    
                    # update next NPU
                    next = (next - 1) % npus_count

        Check()
        XML()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--npus_count', type=int, help ='number of NPUs')
    parser.add_argument('--chunks_per_npu', type=int, help ='initial number of chunks per NPU')
    args = parser.parse_args()
    
    assert args.chunks_per_npu % 2 == 0, "chunks_per_npu must be even"

    # run MSCCLang-DSL to generate MSCCL-IR
    bi_ring_allreduce(args.npus_count, args.chunks_per_npu)


if __name__ == '__main__':
    main()
