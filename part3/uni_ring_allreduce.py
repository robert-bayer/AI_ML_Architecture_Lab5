import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce


def uni_ring_allreduce(npus_count: int) -> None:
    topology = fully_connected(npus_count)
    
    # Collective Definition
    # Note this is All-Reduce, not All-Gather
    # 
    # Number of NPUs (N) = npus_count
    # Number of chunks (C) = npus_count
    # Since this is All-Reduce, all NPU starts with N number of chunks,
    # and also ends with N number of chunks (but each chunks are fully reduced)
    #
    # inplace=True means that Buffer.input == Buffer.output
    #    Hint: Since this is All-Reduce, len(Buffer.input) == len(Buffer.output) == N,
    #    so you can stick to the use of Buffer.input.
    collective = AllReduce(num_ranks=npus_count, chunk_factor=npus_count, inplace=True)

    with MSCCLProgram("uni_ring_allreduce", topology, collective, 1):
        # For every NPU:
        for npu in range(npus_count):
            # this NPU will start sending out chunk (=index) npu.
            chunk_id = npu
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
                
        # Check the correctness of the program
        Check()
        
        # Generate MSCCL-IR
        XML()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--npus_count', type=int, help ='number of NPUs')
    args = parser.parse_args()

    # run MSCCLang-DSL to generate MSCCL-IR
    uni_ring_allreduce(args.npus_count)


if __name__ == '__main__':
    main()
