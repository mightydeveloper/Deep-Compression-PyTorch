import os
from collections import defaultdict, namedtuple
import pickle
from heapq import heappush, heappop, heapify
import struct

import numpy as np
from scipy.sparse import csr_matrix

Node = namedtuple('Node', 'freq value left right')

def huffman_encode(arr, prefix):
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

    def generate_code(node, code):
        if node == None:
            return
        if node.value != False:
            value2code[node.value] = code
            return
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')

    root = heappop(heap)
    generate_code(root, '')

    # Dump data
    data_encoding = ''.join(value2code[float(value)] for value in np.nditer(arr))
    dump(data_encoding, f'{prefix}.bin')

    # Dump codebook (huffman tree)
    codebook_encoding = encode_huffman_tree(root)
    dump(codebook_encoding, f'{prefix}_codebook.bin')

    return root


def huffman_decode(prefix):
    # Read the codebook
    codebook_encoding = load(f'{prefix}_codebook.bin')
    root = decode_huffman_tree(codebook_encoding)

    # Read the data
    data_encoding = load(f'{prefix}.bin')

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value != False: # Leaf node
            data.append(ptr.value)
            ptr = root

    return np.array(data, dtype='float32'), root




# Logics to encode / decode huffman tree
# Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
def encode_huffman_tree(root):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    code_list = []
    def encode_node(node):
        if node.value != False: # node is leaf node
            code_list.append('1')
            code_list.extend(list(float2bitstr(node.value)))
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)
    encode_node(root)
    return ''.join(code_list)


def decode_huffman_tree(code_str):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    idx = 0
    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1': # Leaf node
            value = bitstr2float(code_str[idx:idx+32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, False, left, right)

    return decode_node()



# My own dump / load logics
def dump(code_str, filename):
    """
    code_str : string of either '0' and '1' characters
    this function dumps to a file
    """
    # Make header (1 byte) and add padding to the end
    # Files need to be byte aligned.
    # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
    # This introduces minimum of 8 bits, maximum of 15 bits overhead
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    with open(filename, 'wb') as f:
        f.write(byte_arr)


def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    with open(filename, 'rb') as f:
        header = f.read(1)
        rest = f.read() # bytes
        code_str = ''.join(f'{byte:08b}' for byte in rest)
        code_str = code_str[:-ord(header)] # string of '0's and '1's
    return code_str


# Helper functions for converting between bit string and float
def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]


#def huffman_encode_model(model):
#    names = ['fc1', 'fc2', 'fc3']
#    folder = 'encodings/'
#    os.makedirs(folder, exist_ok=True)
#    for name, module in zip(names, model.children()):
#        weight = module.weight.data.cpu().numpy()
#        csr = csr_matrix(weight)
#        #print('number of csr data :', len(csr.data))
#        #print('number of scr indices :', len(csr.indices))
#        #print('number of scr indptr:', len(csr.indptr))
#        #print('bytes of csr data :', csr.data.nbytes)
#        #print('bytes of csr indices:', csr.indices.nbytes)
#        #print('bytes of csr indptr:', csr.indptr.nbytes)
#        h.compress(csr.data, folder+name+'.bin')
