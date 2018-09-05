import os
from collections import defaultdict, namedtuple
import numpy as np
from heapq import heappush, heappop, heapify
from scipy.sparse import csr_matrix

Node = namedtuple('Node', 'freq value left right')

class HuffmanCoder:
    def __init__(self):
        pass

    def compress(self, arr, output='compressed.bin'):
        # Calculate frequency in arr
        freq_map = defaultdict(int)
        for value in np.nditer(arr):
            freq_map[float(value)] += 1

        # Make heap
        heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
        heapify(heap)

        # Merge nodes
        while(len(heap) > 1):
            node1 = heappop(heap)
            node2 = heappop(heap)
            merged = Node(node1.freq + node2.freq, False, node1, node2)
            heappush(heap, merged)

        # Generate code value mapping
        value2code = {}
        code2value = {}

        def generate_code(node, code):
            if node == None:
                return
            if node.value != False:
                value2code[node.value] = code
                code2value[code] = node.value
                return
            generate_code(node.left, code + '0')
            generate_code(node.right, code + '1')

        root = heappop(heap)
        generate_code(root, '')

        # Convert to code
        code_str = ''.join(value2code[float(value)] for value in np.nditer(arr))

        # Make header (1 byte) and add padding to the end
        # Files need to be byte aligned.
        # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
        # This introduces minimum of 8 bits, maximum of 15 bits overhead
        num_of_padding = -len(code_str) % 8
        header = f"{num_of_padding:08b}"
        code_str = header + code_str + '0' * num_of_padding

        # Convert string to integers and to real bytes
        byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

        with open(output, 'wb') as f:
            f.write(byte_arr)


def huffman_encode(model):
    h = HuffmanCoder()
    names = ['fc1', 'fc2', 'fc3']
    folder = 'encodings/'
    os.makedirs(folder, exist_ok=True)
    for name, module in zip(names, model.children()):
        weight = module.weight.data.cpu().numpy()
        csr = csr_matrix(weight)
        #print('number of csr data :', len(csr.data))
        #print('number of scr indices :', len(csr.indices))
        #print('number of scr indptr:', len(csr.indptr))
        #print('bytes of csr data :', csr.data.nbytes)
        #print('bytes of csr indices:', csr.indices.nbytes)
        #print('bytes of csr indptr:', csr.indptr.nbytes)
        h.compress(csr.data, folder+name+'.bin')







