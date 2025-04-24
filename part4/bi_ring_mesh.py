import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.collectives import *
from msccl.language.collectives import AllGather
from typing import List, Tuple

def allgather_bi_ring_mesh(width: int, height: int) -> None:
    # number of NPUs: width * height
    # Assumption: width and height are even
    npus_count = width * height
    topology = fully_connected(npus_count)
    
    # Note: this is All-Gather, with 2 initial chunks per NPU.
    # i.e., for each NPU, chunk 0 will be processed in the original direction
    # ahd chunk 1 will be processed in the opposite direction
    collective = AllGather(num_ranks=npus_count, chunk_factor=2, inplace=True)

    ### ===============================================
    # Lab 5.4.2
    # TODO: You may copy-paste your Lab 5.4.1 implementation
    def coord_to_id(x: int, y: int) -> int:
        return y * width + x
    
    def get_ring() -> List[int]:
        ring = list()
        x = 0
        y = 0
        right = True
        ring.append(coord_to_id(x, y))

        architechture

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
    ### ===============================================

    with MSCCLProgram("allgather_bi_ring_mesh", topology, collective, 1):
        # get ring
        ring = get_ring()
        
        ### ===============================================
        # Lab 5.4.2
        # TODO: Finish implementing the bidirectional Ring All-Gather algorithm.
        
        # Hint: Modify the implementation of your uni_ring_mesh.py
        # Hint: chunk 0 will be processed in the original direction
        # Hint: whereas chunk 1 will be processed in the opposite direction
        ### ===============================================
                        
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
        
    allgather_bi_ring_mesh(args.width, args.height)


if __name__ == '__main__':
    main()
