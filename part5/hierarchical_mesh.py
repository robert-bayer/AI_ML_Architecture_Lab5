import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllReduce
from typing import List

def hierarchical_mesh(width: int, height: int) -> None:   
    # number of NPUs: width * height
    # Assumption: width and height are even numbers
    npus_count = width * height
    topology = fully_connected(npus_count)
    npu_lists = list()
    
    # Note: this is All-Reduce.
    # Each NPU starts with: N (=npus_count) number of chunks, and ends with N number of chunks.
    collective = AllReduce(num_ranks=npus_count, chunk_factor=npus_count, inplace=True)
    
    ### ===============================================
    # Lab 5.5.2 and 5.5.3 Helper function
    # TODO: You may copy-paste your Lab 5.4.1 implementation
    def coord_to_id(x: int, y: int) -> int:
        return y * width + x
    ### ===============================================

    def uni_ring_all_reduce(npus: List[int]) -> None:
        # print("All Reduce Called")
        ### ===============================================
        # Lab 5.5.1
        # TODO: Implement this function, which executes the unidirectional Ring All-Reduce algorithm
        # TODO: When the list of NPUs is given.
        # e.g., npus = [0, 1, 2, 3, 7, 6, 5, 4] runs the Ring All-Reduce algorithm among 8 NPUs.
        
        # Hint: Although the #NPUs of the above example is 8, the #chunks of the collective is still 16 (=N).
        # Hint: This means each NPU should process 16 / 8 = 2 chunks.
        
        # Hint: Modify your implementation of uni_ring_allreduce_updated.py
        # Hint: But note that the chunks to be processed per NPU is (#N / len(npus))
        # Hint: think of the chunk_id each NPU will process.
        
        # Hint: Also, the next npu is determined by npus[i + 1], not simply (next + 1).
        
        ### ===============================================
        chunks_per_npu = npus_count // len(npus)
        
        for i, npu in enumerate(npus):

            # chunk_start = i * chunks_per_npu

            for ch in range(chunks_per_npu):

                # this NPU will start sending out chunk (=index) npu.
                # chunk_id = chunk_start + ch
                chunk_id = i + (ch * len(npus))
                # print(f"NPU: {npu}\nI: {i}\nChunks Per NPU: {chunks_per_npu}\nChunk Start {chunk_start}\nChunk ID: {chunk_id}\n")
                c = chunk(rank=npu, buffer=Buffer.input, index=chunk_id)
                
                # run Ring Reduce-Scatter
                # each chunk perspective, this process is:
                #   1. Send this chunk to the next NPU (npu + 1)
                #   2. The receiver NPU reduces the chunk with its own chunk
                #   3. Repeat this for (N-1) steps
                next_idx = (i + 1) % len(npus)
                next = npus[next_idx]
                for step in range(len(npus) - 1):
                    # corresponding chunk on the next NPU
                    next_c = chunk(rank=next, buffer=Buffer.input, index=chunk_id)
                    
                    # reduce to this chunk by sending c
                    c = next_c.reduce(c)
                    
                    # Note: now c denotes the reduced chunk on the next NPU (next),
                    # not the original chunk on the sender NPU (npu).
                    
                    # update next NPU
                    next_idx = (next_idx + 1) % len(npus)
                    next = npus[next_idx]

                # likewise, run Ring All-Gather.
                # currently, the final reduced chunk is at NPU (next - 1).
                # send the final reduced chunk to the next NPU
                # which takes N-1 steps
                for step in range(len(npus) - 1):
                    # send the chunk to the next NPU
                    c = c.copy(dst=next, buffer=Buffer.input, index=chunk_id)
                    
                    # update next NPU
                    next_idx = (next_idx + 1) % len(npus)
                    next = npus[next_idx]

    def phase1() -> None:
        ### ===============================================
        # Lab 5.5.2
        # TODO: Implement phase 1 of the hierarchical mesh All-Reduce algorithm.
        
        # Hint: for 4x4 mesh, phase 1 will have two ring All-Reduce operations:
        #   1. Ring All-Reduce among: [0, 1, 2, 3, 7, 6, 5, 4]
        #   2. Ring All-Reduce among: [8, 9, 10, 11, 15, 14, 13, 12]
        
        # Hint: construct the npus list accordingly, and invoke the uni_ring_all_reduce function.

        for y in range(0, height, 2):
            ring = list()
            for x in range(0, width):
                ring.append(coord_to_id(x, y))
            for x in range(width - 1, 0 - 1, -1):
                ring.append(coord_to_id(x, y + 1))
            

            npu_lists.append(ring)


        # print("PHASE ONE")
        for ring in npu_lists:
            # print(ring)
            uni_ring_all_reduce(ring)

        ### ===============================================
    
    def phase2() -> None:
        ### ===============================================
        # Lab 5.5.3
        # TODO: Implement phase 2 of the hierarchical mesh All-Reduce algorithm.
        
        # Hint: like phase 2, think of the npus list and invoke the uni_ring_all_reduce function.
        # Hint: for example, for 4x4 mesh, the npus list will be:
        #      [0, 8], [4, 12], [1, 9], [5, 13], [2, 10], [6, 14], [3, 11], [7, 15]
        
        phase2_rings = list()

        if len(npu_lists[0])/2 - len(npu_lists[0])//2 != 0.0:
            mod = 1
        else:
            mod = 0

        for i in range(len(npu_lists[0])//2):
            gather = list()
            gather2 = list()
            for ring in npu_lists:
                gather.append(ring[i])
                gather2.append(ring[0 - 1 - i])
            phase2_rings.append(gather)
            phase2_rings.append(gather2)

        if mod:
            gather = list()
            for ring in npu_lists:
                gather.append(ring[len(npu_lists[0])//2])
            phase2_rings.append(gather)

        # print("PHASE TWO")
        for ring in phase2_rings:
            # print(ring)
            uni_ring_all_reduce(ring)


        ### ===============================================



    with MSCCLProgram("hierarchical_mesh", topology, collective, 1):
        # run phase 1
        phase1()
        
        # run phase 2
        phase2()
        
        Check()
        XML()

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, help ='width')
    parser.add_argument('--height', type=int, help ='height')
    args = parser.parse_args()

    # run MSCCLang-DSL to generate MSCCL-IR
    assert args.width % 2 == 0, "width must be even"
    assert args.height % 2 == 0, "width must be even"
        
    hierarchical_mesh(args.width, args.height)


if __name__ == '__main__':
    main()
