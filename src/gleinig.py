# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    An Efficient Algorithm for Sparse Quantum State Preparation
    https://ieeexplore.ieee.org/document/9586240
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UGate
from qclib.gates.ldmcu import Ldmcu
from qclib.gates.initialize_sparse import InitializeSparse
from networkx import Graph
import networkx as nx
import matplotlib as plt


class MergeInitialize(InitializeSparse):
    """
    An Efficient Algorithm for Sparse Quantum State Preparation
    https://ieeexplore.ieee.org/document/9586240
    """

    def __init__(self, params, label=None):
        """
        Classical algorithm that creates a quantum circuit C that loads
        a sparse quantum state, applying a sequence of operations maping
        the desired state |sigma> to |0>. And then inverting C to obtain
        the mapping of |0> to the desired state |sigma>.
        Args:
        params: A dictionary with the non-zero amplitudes corresponding to each state in
                    format { '000': <value>, ... , '111': <value> }
        Returns:
        Creates a quantum gate that maps |0> to the desired state |params>
        """

        self._name = "merge"

        if label is None:
            label = "MERGESP"

        # Parameters need to be validated first by superclass
        self._get_num_qubits(params)

        super().__init__(self._name, self.num_qubits, params.items(), label=label)
        self.original_basis = list(params.keys())
        self.transformed_basis = list(params.keys())
        self.path = []

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        state_dict = dict(self.params)
        b_strings = list(state_dict.keys())

        n_qubits = len(b_strings[0])
        quantum_register = QuantumRegister(n_qubits)
        quantum_circuit = QuantumCircuit(quantum_register)

        while len(b_strings) > 1:
            bitstr1, bitstr2, dif, dif_qubits = self._select_strings(state_dict)

            bitstr1, bitstr2, state_dict, quantum_circuit = self._preprocess_states(
                bitstr1, bitstr2, dif, dif_qubits, state_dict, quantum_circuit
            )

            state_dict, quantum_circuit = self._merge(
                state_dict, quantum_circuit, bitstr1, bitstr2, dif_qubits, dif
            )
            b_strings = list(state_dict.keys())

            idx1 = self.transformed_basis.index(bitstr1)
            idx2 = self.transformed_basis.index(bitstr2)
            self.path.append([self.original_basis[idx1], self.original_basis[idx2]])

            # print("transformed basis: ", self.transformed_basis)

        b_string = b_strings.pop()
        for (bit_idx, bit) in enumerate(b_string):
            if bit == "1":
                quantum_circuit.x(bit_idx)

        # bases = list(reversed(self.path))
        # print(bases)
        # fig = plt.figure()
        # graph = Graph()
        # for pair in bases:
        #     val = sum([b1 != b2 for b1, b2 in zip(pair[0], pair[1])])
        #     graph.add_edge(pair[0], pair[1], weight=val)
        # graph.graph["start"] = bases[0][0]
        # labels = nx.get_edge_attributes(graph, 'weight')
        # pos = nx.spring_layout(graph)
        # nx.draw(graph, pos, with_labels=True, node_color="lightblue")
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        # Save the figure to a file
        # plt.pyplot.savefig(f"graph_{bases[0]}.png")
        # plt.pyplot.clf()
        # quantum_circuit.reverse_ops().draw(output="mpl", fold=-1, filename=f"gleinig_{bases[0]}.jpg")

        return quantum_circuit.reverse_ops()

    @staticmethod
    def initialize(q_circuit, state, qubits=None):
        if qubits is None:
            q_circuit.append(MergeInitialize(state), q_circuit.qubits)
        else:
            q_circuit.append(MergeInitialize(state), qubits)

    @staticmethod
    def _maximizing_difference_bit_search(b_strings, dif_qubits):
        """
        Splits the set of bit strings into two (t_0 and t_1), by setting
        t_0 as the set of bit_strings with 0 in the bit_index position, and
        t_1 as the set of bit_strings with 1 in the bit_index position.
        Searching for the bit_index not in dif_qubits that maximizes the difference
        between the size of the nonempty t_0 and t_1.
        Args:
        b_string: A list of bit strings eg.: ['000', '011', ...,'101']
        dif_qubits: A list of previous qubits found to maximize the difference
        Returns:
        bit_index: The qubit index that maximizes abs(len(t_0)-len(t_1))
        t_0: List of binary strings with 0 on the bit_index qubit
        t_1: List of binary strings with 1 on the bit_index qubit
        """
        t_0 = []
        t_1 = []
        bit_index = 0
        set_difference = -1
        bit_search_space = list(set(range(len(b_strings[0]))) - set(dif_qubits))

        for bit in bit_search_space:
            temp_t0 = [x for x in b_strings if x[bit] == "0"]
            temp_t1 = [x for x in b_strings if x[bit] == "1"]

            if temp_t0 and temp_t1:
                temp_difference = np.abs(len(temp_t0) - len(temp_t1))
                if temp_difference > set_difference:
                    t_0 = temp_t0
                    t_1 = temp_t1
                    bit_index = bit
                    set_difference = temp_difference

        return bit_index, t_0, t_1

    @staticmethod
    def _build_bit_string_set(b_strings, dif_qubits, dif_values):
        """
        Creates a new set of bit strings from b_strings, where the bits
        in the indexes in dif_qubits match the values in dif_values.

        Args:
        b_strings: list of bit strings eg.: ['000', '011', ...,'101']
        dif_qubits: list of integers with the bit indexes
        dif_values: list of integers values containing the values each bit
                    with index in dif_qubits shoud have
        Returns:
        A new list of bit_strings, with matching values in dif_values
        on indexes dif_qubits
        """
        bit_string_set = []
        for b_string in b_strings:
            if [b_string[i] for i in dif_qubits] == dif_values:
                bit_string_set.append(b_string)

        return bit_string_set

    def _bit_string_search(self, b_strings, dif_qubits, dif_values):
        """
        Searches for the bit strings with unique qubit values in `dif_values`
        on indexes `dif_qubits`.
        Args:
        b_strings: List of binary strings where the search is to be performed
                    e.g.: ['000', '010', '101', '111']
        dif_qubits: List of indices on a binary string of size N e.g.: [1, 3, 5]
        dif_values: List of values each qubit must have on indexes stored in dif_qubits [0, 1, 1]
        Returns:
        b_strings: One size list with the string found, to have values dif_values on indexes
                    dif_qubits
        dif_qubits: Updated list with new indexes
        dif_values: Updated list with new values
        """
        temp_strings = b_strings
        while len(temp_strings) > 1:
            bit, t_0, t_1 = self._maximizing_difference_bit_search(
                temp_strings, dif_qubits
            )
            dif_qubits.append(bit)
            if len(t_0) < len(t_1):
                dif_values.append("0")
                temp_strings = t_0
            else:
                dif_values.append("1")
                temp_strings = t_1

        return temp_strings, dif_qubits, dif_values

    def _select_strings(self, state_dict):
        """
        Searches for the states described by the bit strings bitstr1 and bitstr2 to be merged
        Args:
        state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                    binary strings as keys e.g.: {'001': <value>, '101': <value>}
        Returns:
        bitstr1: First binary string
        bitstr2: Second binary string
        dif_qubit: Qubit index to be used as target for the merging operation
        dif_qubits: List of qubit indexes where bitstr1 and bitstr2 must be equal, because the
                    correspondig qubits of those indexes are to be used as control for the
                    merging operation
        """
        # Initialization
        dif_qubits = []
        dif_values = []
        b_strings1 = b_strings2 = list(state_dict.keys())

        # Searching for bitstr1
        (b_strings1, dif_qubits, dif_values) = self._bit_string_search(
            b_strings1, dif_qubits, dif_values
        )
        dif_qubit = dif_qubits.pop()
        dif_values.pop()
        bitstr1 = b_strings1[0]

        # Searching for bitstr2
        b_strings2.remove(bitstr1)
        b_strings1 = self._build_bit_string_set(b_strings2, dif_qubits, dif_values)
        (b_strings1, dif_qubits, dif_values) = self._bit_string_search(
            b_strings1, dif_qubits, dif_values
        )
        bitstr2 = b_strings1[0]

        return bitstr1, bitstr2, dif_qubit, dif_qubits

    @staticmethod
    def _apply_operation_to_bit_string(b_string, operation, qubit_indexes):
        """
        Applies changes on binary strings according to the operation
        Args:
        b_string: Binary string '00110'
        operation: Operation to be applied to the string
        qubit_indexes: Indexes of the qubits on the binary strings where the operations are to
                        be applied
        Returns:
        Updated binary string
        """
        assert operation in ["x", "cx"]

        if operation == "x":
            compute = _compute_op_x

        else:
            compute = _compute_op_cx

        return compute(b_string, qubit_indexes)

    @staticmethod
    def _update_state_dict_according_to_operation(
            state_dict, operation, qubit_indexes, merge_strings=None
    ):
        """
        Updates the keys of the state_dict according to the operation being applied to the circuit
        Args:
        state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                    binary strings as keys e.g.: {'001': <value>, '101': <value>}
        operation: Operation to be applied to the states, it must be ['x', 'cx', 'merge']
        qubit_indexes: Indexes of the qubits on the binary strings where the operations are to
                        be applied
        merge_strings: Binary strings associated ot the states on the quantum processor
                        to be merge e.g.:['01001', '10110']
        Returns:
        A state_dict with the updated states
        """
        assert operation in ["x", "cx", "merge"]
        state_list = list(state_dict.items())
        new_state_dict = {}
        if operation == "merge":
            assert merge_strings is not None
            # Computes the norm of bitstr1 and bitstr2
            new_state_dict = state_dict.copy()
            norm = np.linalg.norm(
                [new_state_dict[merge_strings[0]], new_state_dict[merge_strings[1]]]
            )
            new_state_dict.pop(merge_strings[1], None)
            new_state_dict[merge_strings[0]] = norm
        else:
            for (bit_string, value) in state_list:
                temp_bstring = MergeInitialize._apply_operation_to_bit_string(
                    bit_string, operation, qubit_indexes
                )
                new_state_dict[temp_bstring] = value

        return new_state_dict

    # @staticmethod
    def _equalize_bit_string_states(self,
                                    bitstr1, bitstr2, dif, state_dict, quantum_circuit
                                    ):
        """
        Applies operations to the states represented by bit strings bitstr1 and bitstr2 equalizing
        them at every qubit except the one in the dif index. And alters the bit strings and
        state_dict accordingly.
        Args:
        bitstr1: First bit string
        bitstr2: Second bit string
        dif: index where both bitstr1 and bitstr2 must be different
        state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                    binary strings as keys e.g.: {'001': <value>, '101': <value>}
        quantum_circuit: Qiskit's quantum circuit's object with the gates applied to the circuit
        Returns:
        Updated bitstr1, bitstr2, state_dict and quantum_circuit
        """
        b_index_list = list(range(len(bitstr1)))
        b_index_list.remove(dif)

        for b_index in b_index_list:
            if bitstr1[b_index] != bitstr2[b_index]:
                quantum_circuit.cx(dif, b_index)
                bitstr1 = MergeInitialize._apply_operation_to_bit_string(
                    bitstr1, "cx", [dif, b_index]
                )
                bitstr2 = MergeInitialize._apply_operation_to_bit_string(
                    bitstr2, "cx", [dif, b_index]
                )
                state_dict = MergeInitialize._update_state_dict_according_to_operation(
                    state_dict, "cx", [dif, b_index]
                )
                self.transformed_basis = [self._apply_operation_to_bit_string(z1, "cx", [dif, b_index]) for z1 in self.transformed_basis]

        return bitstr1, bitstr2, state_dict, quantum_circuit

    # @staticmethod
    def _apply_not_gates_to_qubit_index_list(self,
                                             bitstr1, bitstr2, dif_qubits, state_dict, quantum_circuit
                                             ):
        """
        Applies quantum not gate at the qubit at a given index, where the state represented by the
        bit string bitstr2 is different than '1' at index in diff_qubits.
        Args:
        bitstr1: First bit string
        bitstr2: Second bit string
        dif_qubits: indexes where both bitstr1 and bitstr2 are equal
        state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                    binary strings as keys e.g.: {'001': <value>, '101': <value>}
        quantum_circuit: Qiskit's quantum circuit's object with the gates applied to the circuit
        Returns:
        Updated bitstr1, bitstr2, state_dict and quantum_circuit
        """
        for b_index in dif_qubits:
            if bitstr2[b_index] != "1":
                quantum_circuit.x(b_index)
                bitstr1 = MergeInitialize._apply_operation_to_bit_string(bitstr1, "x", b_index)
                bitstr2 = MergeInitialize._apply_operation_to_bit_string(bitstr2, "x", b_index)
                state_dict = MergeInitialize._update_state_dict_according_to_operation(
                    state_dict, "x", b_index
                )
                self.transformed_basis = [self._apply_operation_to_bit_string(z1, "x", b_index) for z1 in self.transformed_basis]
        return bitstr1, bitstr2, state_dict, quantum_circuit

    def _preprocess_states(
            self, bitstr1, bitstr2, dif, dif_qubits, state_dict, quantum_circuit
    ):
        """
        Apply the operations on the basis states to prepare for merging bitstr1 and bitstr2.
        Args:
        state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                    binary strings as keys e.g.: {'001': <value>, '101': <value>}
        bitstr1: First binary string to be merged
        bitstr2: Second binary string to be merged
        dif_qubits: List of qubit indexes on the binary strings
        dif: Target qubit index where the merge operation is to be applied
        quantum_circuit: Qiskit's QuantumCircuit object where the operations are to be called
        Returns:
        state_dict: Updated state dict
        bitstr1: First updated binary string to be merged
        bitstr2: Second updated binary string to be merged
        quantum_circuit: Qiskit's quantum circuit's object with the gates applied to the circuit
        """

        if bitstr1[dif] != "1":
            quantum_circuit.x(dif)
            bitstr1 = MergeInitialize._apply_operation_to_bit_string(bitstr1, "x", dif)
            bitstr2 = MergeInitialize._apply_operation_to_bit_string(bitstr2, "x", dif)
            state_dict = self._update_state_dict_according_to_operation(
                state_dict, "x", dif
            )
            self.transformed_basis = [self._apply_operation_to_bit_string(z1, "x", dif) for z1 in self.transformed_basis]

        (
            bitstr1,
            bitstr2,
            state_dict,
            quantum_circuit,
        ) = self._equalize_bit_string_states(
            bitstr1, bitstr2, dif, state_dict, quantum_circuit
        )

        (
            bitstr1,
            bitstr2,
            state_dict,
            quantum_circuit,
        ) = self._apply_not_gates_to_qubit_index_list(
            bitstr1, bitstr2, dif_qubits, state_dict, quantum_circuit
        )

        return bitstr1, bitstr2, state_dict, quantum_circuit

    @staticmethod
    def _compute_angles(amplitude_1, amplitude_2):
        """
        Computes the angles for the adjoint of the merge matrix M
        that is going to map the dif qubit to zero e.g.:
        M(a|0> + b|1>) -> |1>

        Args:
        amplitude_1: A complex/real value, associated with the string with
                        1 on the dif qubit
        amplitude_2: A complex/real value, associated with the string with
                        0 on the dif qubit
        Returns:
        The angles theta, lambda and phi for the U operator
        """
        norm = np.linalg.norm([amplitude_1, amplitude_2])

        phi = 0
        lamb = 0
        # there is no minus on the theta because the intetion is to compute the inverse
        if isinstance(amplitude_1, complex) or isinstance(amplitude_2, complex):
            amplitude_1 = (
                complex(amplitude_1)
                if not isinstance(amplitude_1, complex)
                else amplitude_1
            )
            amplitude_2 = (
                complex(amplitude_2)
                if not isinstance(amplitude_2, complex)
                else amplitude_2
            )

            theta = -2 * np.arcsin(np.abs(amplitude_2 / norm))
            lamb = np.log(amplitude_2 / norm).imag
            phi = np.log(amplitude_1 / norm).imag - lamb

        else:
            theta = -2 * np.arcsin(amplitude_2 / norm)

        return theta, phi, lamb

    def _merge(self, state_dict, quantum_circuit, bitstr1, bitstr2, dif_qubits, dif):

        theta, phi, lamb = self._compute_angles(
            state_dict[bitstr1], state_dict[bitstr2]
        )

        # Applying merge operation
        merge_gate = UGate(theta, phi, lamb, label="U")
        if not dif_qubits:
            quantum_circuit.append(merge_gate, dif_qubits + [dif], [])
        else:
            gate_definition = UGate(theta, phi, lamb, label="U").to_matrix()
            Ldmcu.ldmcu(quantum_circuit, gate_definition, dif_qubits, dif)

        state_dict = self._update_state_dict_according_to_operation(
            state_dict, "merge", None, merge_strings=[bitstr1, bitstr2]
        )

        return state_dict, quantum_circuit


def _compute_op_cx(xlist, idx):
    return (
        f'{xlist[: idx[1]]}{(not int(xlist[idx[1]])) * 1}{xlist[idx[1] + 1:]}'
        if xlist[idx[0]] == "1"
        else xlist
    )


def _compute_op_x(xlist, idx):
    return (
        xlist[:idx] + "1" + xlist[idx + 1:]
        if xlist[idx] == "0"
        else xlist[0:idx] + "0" + xlist[idx + 1:]
    )