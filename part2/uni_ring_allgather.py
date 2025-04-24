import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllGather


def uni_ring_allgather(npus_count: int) -> None:
    topology = fully_connected(npus_count)
    
    # Collective Definition
    #
    # Number of NPUs (N) = npus_count
    # Number of chunks (C) = 1 per NPU
    #
    # Since this is All-Gather, all NPU starts with 1 chunk, and ends with N chunks.
    #
    # inplace=True means that Buffer.input == Buffer.output
    #   Caveat: still, MSCCLang thinks len(Buffer.input) == 1, len(Buffer.output) == N
    collective = AllGather(num_ranks=npus_count, chunk_factor=1, inplace=True)

    with MSCCLProgram("uni_ring_allgather", topology, collective, 1):
        # For every NPU:
        for npu in range(npus_count):
            # This NPU starts with chunk at Buffer.input, whose ID (=index) is 0
            c = chunk(rank=npu, buffer=Buffer.input, index=0)
            
            # We should run Ring All-Gather to broadcast this chunk to all other NPUs.
            # Chunk perspective, this process is:
            #   1. Send this chunk to the next NPU (npu + 1)
            #   2. The receiver NPU repeats this process
            #   3. For (N-1) steps
            next = (npu + 1) % npus_count
            for step in range(npus_count - 1):
                # send the chunk to the next NPU
                c = c.copy(dst=next, buffer=Buffer.output, index=npu)
                
                # Note: now c denotes the received chunk on the next NPU (next),
                # not the original chunk on the sender NPU (npu).
                
                # update next NPU to repeat the process
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
    uni_ring_allgather(args.npus_count)


if __name__ == '__main__':
    main()
