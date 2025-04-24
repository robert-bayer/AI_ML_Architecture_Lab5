import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllGather
from typing import List, Tuple

def allgather_uni_ring_mesh(width: int, height: int) -> None:
    # Number of NPUs: width * height
    npus_count = width * height
    topology = fully_connected(npus_count)
    
    # Note: this is All-Gather, with 1 chunk per NPU.
    collective = AllGather(num_ranks=npus_count, chunk_factor=1, inplace=True)
    
    def coord_to_id(x: int, y: int) -> int:
        ### ===============================================
        # Lab 5.4.1 helper function
        # TODO: Implement this helper function
        # TODO: which converts 2D coordinates to NPU id
        
          # remove this
        ### ===============================================
        return y * width + x
    
    def get_ring() -> List[int]:
        ring = list()
           
        
        ### ===============================================
        # Lab 5.4.1
        # TODO: Implement this helper function which returns the NPUs of the ring.
        
        # Hint: for a 2x2 mesh, the ring is
        #       [0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4]
        
        # Hint: Start from 0
        # Hint: then, as necessary, repeat the following steps:
        #   go right until the end of the row
        #   go down to the next row
        #   go left until the end of the row
        # Hint: then, move up to finish the ring
        ### ===============================================

        x = 0
        y = 0
        right = True
        ring.append(coord_to_id(x, y))

        while (coord_to_id(x, y) != coord_to_id(1, height - 1)):
            if right:
                if ((coord_to_id(x, y) % 4) == (width - 1)):
                    y += 1
                    right = False
                else :
                    x += 1
            else:
                if (coord_to_id(x, y) % 4 == 1):
                    y += 1
                    right = True
                else :
                    x -= 1
            ring.append(coord_to_id(x, y))

        for idx in range(coord_to_id(0, height - 1), 0, -width):
            ring.append(idx)
        
        return ring

    with MSCCLProgram("allgather_uni_ring_mesh", topology, collective, 1):
        # get ring
        ring = get_ring()
        
        ### ===============================================
        # Lab 5.4.1
        # TODO: Finish implementing the All-Gather algorithm.
        
        # Hint: You may modify the implementation of uni_ring_allgather.sh,
        # Hint: but instead of simple next = next + 1, you'll to use the `ring`.
        ### ===============================================

        for itx in range(npus_count):
            npu = ring[itx]
            next_itx = (itx + 1) % npus_count
            # This NPU starts with chunk at Buffer.input, whose ID (=index) is 0
            c = chunk(rank=npu, buffer=Buffer.input, index=0)
            
            # We should run Ring All-Gather to broadcast this chunk to all other NPUs.
            # Chunk perspective, this process is:
            #   1. Send this chunk to the next NPU (npu + 1)
            #   2. The receiver NPU repeats this process
            #   3. For (N-1) steps
            next = ring[next_itx]
            for step in range(npus_count - 1):
                # send the chunk to the next NPU
                c = c.copy(dst=next, buffer=Buffer.output, index=npu)
                
                # Note: now c denotes the received chunk on the next NPU (next),
                # not the original chunk on the sender NPU (npu).
                
                # update next NPU to repeat the process
                next_itx = (next_itx + 1) % npus_count
                next = ring[next_itx]
                
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
        
    allgather_uni_ring_mesh(args.width, args.height)


if __name__ == '__main__':
    main()
