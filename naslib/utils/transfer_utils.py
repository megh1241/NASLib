import hashlib
import collections

def remove_suffix(string):
    split_string = string.split('.')
    if 'cell' in string:
        return '.'.join([split_string[0] , split_string[1]])
    return split_string[0]

def get_hashed_names(graph, arch):
    hashed_names = collections.defaultdict(str)
    counts = collections.defaultdict(int)

    name_module_dict = {}
    for name, module in arch.named_modules():
        name_module_dict[name] = str(module)

    for name, param in arch.named_parameters():
        name_without_suffix = remove_suffix(name)
        module = name_module_dict[name_without_suffix]
        if name == '':
            continue
        layer_hash = hashlib.sha3_512()
        layer_hash.update(str(module).encode())
        for pred_name in graph.predecessors(name_without_suffix):
            layer_hash.update(hashed_names[pred_name].encode())
        layer_hash = layer_hash.hexdigest()
        base_name = layer_hash + str(module)
        hashed_names[name_without_suffix] = base_name + "_" + str(counts[base_name])
        counts[base_name] += 1
    return hashed_names

'''
def get_hashed_names(graph, arch):
    hashed_names = collections.defaultdict(str)
    counts = collections.defaultdict(int)
    for name, module in arch.named_modules():
        if name == '':
            continue
        layer_hash = hashlib.sha3_512()
        layer_hash.update(str(module).encode())
        name_without_suffix = remove_suffix(name)
        if name_without_suffix not in graph:
            continue
        for pred_name in graph.predecessors(name_without_suffix):
            layer_hash.update(hashed_names[pred_name].encode())
        layer_hash = layer_hash.hexdigest()
        base_name = layer_hash + str(module)
        hashed_names[name_without_suffix] = base_name + "_" + str(counts[base_name])
        counts[base_name] += 1
    return hashed_names
'''

def process_pred_graph(hashed_names, digraph):
    new_graph = {}
    for node in digraph.nodes:
        if hashed_names[node] not in new_graph:
            new_graph[hashed_names[node]] = []
        neighbors = list(digraph.predecessors(node))
        new_graph[hashed_names[node]].extend(list(set([hashed_names[i] for i in neighbors])))
    return new_graph
