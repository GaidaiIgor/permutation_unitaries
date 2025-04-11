from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from src.gleinig import MergeInitialize
from src.permutation_circuit_generator import PermutationCircuitGenerator
from src.permutation_generator import DensePermutationGenerator
from src.utilities.validation import get_state_vector


class StateCircuitGenerator(ABC):
    """ Base class for generating circuits that prepare a given state. """

    @abstractmethod
    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        """ Generates a quantum circuit that prepares target_state, described as dictionary of bitstrings and corresponding probability amplitudes. """
        pass


class MergingStatesGenerator(StateCircuitGenerator):
    """ Generates state preparation circuits via merging states method of Gleinig. """

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        merger = MergeInitialize(target_state)
        circuit = merger._define_initialize()
        return circuit.reverse_bits()


class QiskitDefaultGenerator(StateCircuitGenerator):
    """ Generates state preparation circuit via qiskit's default built-in method. """

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        target_state_vector = get_state_vector(target_state)
        num_qubits = len(next(iter(target_state.keys())))
        circuit = QuantumCircuit(num_qubits)
        circuit.prepare_state(target_state_vector)
        return circuit


@dataclass(kw_only=True)
class DensePermuteGenerator(StateCircuitGenerator):
    """ Generates a circuit for a dense state, then permutes it to the target state. """
    permutation_generator: DensePermutationGenerator
    permutation_circuit_generator: PermutationCircuitGenerator

    @staticmethod
    def map_to_dense_state(state: dict[str, complex], dense_permutation: dict[str, str], dense_qubits: list[int]) -> list[complex]:
        """ Permutes state according to given dense permutation and returns contiguous list of amplitudes where i-th element corresponds to basis i. """
        dense_state = [0] * 2 ** len(dense_qubits)
        for basis, amplitude in state.items():
            mapped_basis = dense_permutation[basis]
            dense_coords = ''.join([mapped_basis[i] for i in dense_qubits])
            ind = int(dense_coords, 2)
            dense_state[ind] = amplitude
        return dense_state

    @abstractmethod
    def get_dense_state_circuit(self, dense_state: list[complex]) -> QuantumCircuit:
        pass

    def generate_circuit(self, target_state: dict[str, complex]) -> QuantumCircuit:
        dense_permutation, dense_qubits = self.permutation_generator.get_permutation(target_state)
        dense_state = self.map_to_dense_state(target_state, dense_permutation, dense_qubits)
        dense_state_qc = self.get_dense_state_circuit(dense_state)
        inverse_permutation = {val: key for key, val in dense_permutation.items()}
        permutation_qc = self.permutation_circuit_generator.get_permutation_circuit(inverse_permutation)
        overall_qc = QuantumCircuit(permutation_qc.num_qubits)
        any_dense_basis = next(iter(inverse_permutation))
        sparse_qubits = list(set(range(len(any_dense_basis))) - set(dense_qubits))
        for qubit in sparse_qubits:
            if any_dense_basis[qubit] == '1':
                overall_qc.x(len(any_dense_basis) - qubit - 1)
        overall_qc.append(dense_state_qc, list(len(any_dense_basis) - 1 - np.array(dense_qubits)[::-1]))
        overall_qc.append(permutation_qc, range(permutation_qc.num_qubits))
        return overall_qc


@dataclass(kw_only=True)
class QiskitDenseGenerator(DensePermuteGenerator):
    """ Uses qiskit's built-in state preparation on dense state. """

    def get_dense_state_circuit(self, dense_state: list[complex]) -> QuantumCircuit:
        """ Returns a quantum circuit that prepares a dense state via qiskit's prepare_state method. """
        num_qubits = int(np.ceil(np.log2(len(dense_state))))
        qc = QuantumCircuit(num_qubits)
        qc.prepare_state(dense_state, range(num_qubits))
        return qc
