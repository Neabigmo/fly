"""flynum — numerosity and addition in a connectome-constrained Drosophila network.

Pipeline stages
---------------
0. data   : download MaleCNS v1.0, build edge list, extract subgraph, shuffle control
1. retina : fixed retinotopic encoder (hex lattice -> 32x32 image sampling)
2. stimuli: dot-count and dot-addition image generators with visual controls
3. model  : leaky recurrent network with frozen topology and trainable per-edge gains
4. train  : BPTT (gains + readout) and linear-probe (frozen fly)
5. analysis: figures, temporal decoding, lesions, report
"""

__version__ = "0.1.0"
