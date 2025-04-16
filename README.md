# Permutation Unitaries

This repository contains implementation of several quantum computing subroutines, related data and figures. 
Specifically, the main contributions are

1. A decomposition algorithm for sparse amplitude permutation unitaries into a product of multi-controlled X (MCX) gates.

    - The algorithm accepts a list of $m$ non-zero amplitude labels on $n$ qubits and their corresponding destinations after permutation.
    
    - It outputs a Qiskit quantum circuit that implements the specified permutation.
    
    - Other (non-specified) amplitudes are assumed to be 0, and can be arbitrarily permuted among themselves.

    - Implemented by `PermutationCircuitGeneratorCluster` class.

2. A sparse clustered quantum state preparation algorithm.

    - Uses a dense state preparation method (Qiskit's `prepapre_state` in this case) to prepare a dense permutation of a given sparse state, 
      then applies a permutation circuit to move the amplitudes to their target destinations. 

    - Works well for sparse clustered states, i.e. states such that many amplitudes are 0 and 
      among the non-zero amplitudes there are many pairs of amplitudes with Hamming distance = 1.

    - Implemented by `QiskitDenseGenerator` class.

See `run_prepare_state` function for an entry point and an example of how to use these classes.

The corresponding publication is available at https://arxiv.org/abs/2504.08705
