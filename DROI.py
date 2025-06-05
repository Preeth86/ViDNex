import pickle
import requests
from pulp import *
import networkx as nx
import json
import sys
import heapq
from scipy.stats import norm
from math import exp
import numpy as np
from networkx.exception import NetworkXNoPath
import matplotlib.pyplot as plt
import matplotlib
import random
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
matplotlib.use('TkAgg')

RYU_IP = '127.0.0.1'  # Adjust IP if Ryu is hosted elsewhere
RYU_PORT = 8080  # Default port for Ryu REST API

output = []

# Custom print function to capture output
def custom_print(*args):
    message = ' '.join(str(arg) for arg in args)
    print(message)
    output.append([message])

# Visualization function to debug graph connectivity
def visualize_graph(graph):
    plt.figure(figsize=(12, 8))
    nx.draw(graph, with_labels=True, node_color='skyblue', node_size=500)
    # plt.title("Substrate Network Graph")
    plt.savefig("sn_graph.png")  # Save the graph as an image
    # plt.show()

# Load SN graph directly from JSON file
def load_sn_graph(sn_json):
    try:
        graph = nx.Graph()
        servers = {}

        for node_id, node_data in sn_json.items():
            if node_id.startswith('h'):
                servers[node_id] = {
                    'cpu': node_data.get('allocated_cores', 0),
                    'original_cpu': node_data.get('original_cores', 0),
                    'vms': []
                }
                graph.add_node(node_id, type='host', cpu=node_data.get('allocated_cores', 0))
            elif node_id.startswith('l') or node_id.startswith('s'):
                graph.add_node(node_id, type='switch')

        for link in sn_json.get('links_details', []):
            node1, node2 = link.get('node1'), link.get('node2')
            bandwidth = link.get('assigned_bandwidth', 0)
            graph.add_edge(node1, node2, bandwidth=bandwidth)

        graph.add_node('c0', type='controller')
        for node in graph.nodes:
            if node.startswith('s'):
                graph.add_edge('c0', node, bandwidth=10000)

        link_flags = {edge: False for edge in graph.edges()}
        visualize_graph(graph)
        return servers, graph, link_flags
    except KeyError as e:
        raise ValueError(f"Invalid SN JSON format. Missing key: {e}")

def calculate_weight(data_i, data_j):
    weight = abs(data_i['allocated_cores'] - data_j['allocated_cores'])
    return weight

def construct_weight_matrix(substrate_data, node_list):
    num_nodes = len(node_list)
    weight_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = calculate_weight(substrate_data[node_list[i]], substrate_data[node_list[j]])
            weight_matrix[i, j] = weight
            weight_matrix[j, i] = weight

    return weight_matrix

def laplacian_matrix(weight_matrix):
    degree_matrix = np.diag(np.sum(weight_matrix, axis=1))
    laplacian = degree_matrix - weight_matrix
    return laplacian

def normalize_laplacian(laplacian, degree_matrix):
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    normalized_laplacian = np.dot(np.dot(d_inv_sqrt, laplacian), d_inv_sqrt)
    return normalized_laplacian

def partition_substrate_network(substrate_data, k1):
    node_list = [node for node in substrate_data.keys() if
                 isinstance(substrate_data[node], dict) and 'allocated_cores' in substrate_data[node]]
    weight_matrix = construct_weight_matrix(substrate_data, node_list)
    laplacian = laplacian_matrix(weight_matrix)
    degree_matrix = np.diag(np.sum(weight_matrix, axis=1))
    norm_laplacian = normalize_laplacian(laplacian, degree_matrix)

    eigvals, eigvecs = eigsh(norm_laplacian, k=k1, which='SM')

    kmeans = KMeans(n_clusters=k1)
    clusters = kmeans.fit_predict(eigvecs)

    resources = [
        np.sum([substrate_data[node_list[i]]['allocated_cores'] for i in range(len(node_list)) if clusters[i] == j])
        for j in range(k1)]
    croi = np.argmax(resources)

    return clusters, croi, node_list

def node_embedding_and_mapping(weight_matrix, virtual_data, substrate_data, clusters, croi, node_list, graph, servers):
    custom_print(f"\nNode Embedding and Mapping of VMs for VNR ID: {virtual_data['vnr_id']}")

    vm_to_server_assignments = {}
    vnr_to_server_assignments = {}

    allocated_hosts = {host: False for host in node_list}
    node_degrees = dict(graph.degree(node_list)) if graph else {node: 0 for node in node_list}

    vnr_id = virtual_data['vnr_id']
    vm_cores = virtual_data['vm_cpu_cores']

    if len(vm_cores) > len(node_list):
        custom_print(f"Mapping unsuccessful: More VMs ({len(vm_cores)}) than available substrate nodes ({len(node_list)})")
        return {}, {}, servers

    sorted_hosts = sorted(node_list, key=lambda x: (clusters[node_list.index(x)] == croi, node_degrees[x]), reverse=True)

    for vm_index, vm_core in enumerate(vm_cores, start=1):
        mapped = False
        for host in sorted_hosts:
            if not allocated_hosts[host] and substrate_data[host]['allocated_cores'] >= vm_core:
                vm_to_server_assignments[f"VM{vm_index}"] = host
                vnr_to_server_assignments.setdefault(vnr_id, []).append(host)

                allocated_hosts[host] = True
                substrate_data[host]['allocated_cores'] -= vm_core
                servers[host]['cpu'] -= vm_core
                servers[host]['vms'].append({'vnr_id': vnr_id, 'vm_index': vm_index, 'cpu': vm_core})
                mapped = True

                custom_print(f"VM{vm_index} mapped to {host} with {vm_core} CPU")
                break

        if not mapped:
            custom_print(f"Mapping unsuccessful: No suitable host found for VM {vm_index}. Rolling back.")
            rollback_embedding(vm_to_server_assignments, [], servers, None)
            return {}, {}, servers

    return vm_to_server_assignments, vnr_to_server_assignments, servers

def dijkstra_shortest_path(graph, start, end):
    """Finds the shortest path between two nodes using Dijkstra’s algorithm, ensuring non-negative weights."""
    try:
        # Create a new weight dictionary where negative bandwidths are ignored
        valid_edges = {(u, v): data["bandwidth"] if data["bandwidth"] > 0 else float("inf")
            for u, v, data in graph.edges(data=True)}

        return nx.shortest_path(graph, source=start, target=end, weight=lambda u, v, d: valid_edges.get((u, v), float("inf")))
    except NetworkXNoPath:
        return None

def link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags, servers):
    custom_print(f"\nLink Embedding and Mapping of VLs for VNR ID: {vnr['vnr_id']}")
    embedding_success = {vnr['vnr_id']: True}
    path_mappings = []

    for link_index, (vm_source, vm_target) in enumerate(vnr['vm_links'], start=1):
        bandwidth_demand = vnr['bandwidth_values'][link_index - 1]

        if f"VM{vm_source}" not in vm_to_server_assignments or f"VM{vm_target}" not in vm_to_server_assignments:
            custom_print(f"Failed to find server assignments for VM{vm_source} or VM{vm_target}. Rolling back node embedding for VNR {vnr['vnr_id']}.")
            rollback_embedding(vm_to_server_assignments, [], servers, None)  # Rollback node embedding
            return {}, graph, []

        source_server = vm_to_server_assignments[f"VM{vm_source}"]
        target_server = vm_to_server_assignments[f"VM{vm_target}"]

        shortest_path = dijkstra_shortest_path(graph, source_server, target_server)

        if shortest_path:
            # Check if bandwidth is sufficient for all links in the selected path
            insufficient_bandwidth = any(
                graph[shortest_path[i]][shortest_path[i + 1]]['bandwidth'] < bandwidth_demand
                for i in range(len(shortest_path) - 1)
            )

            if insufficient_bandwidth:
                custom_print(f"Insufficient bandwidth for embedding link from VM{vm_source} to VM{vm_target}. Rolling back node and link embedding for VNR {vnr['vnr_id']}.")
                rollback_embedding(vm_to_server_assignments, path_mappings, servers, graph)
                return {}, graph, []

            # Deduct bandwidth after successful validation
            path_mappings.append(((source_server, target_server, vnr['vnr_id']), shortest_path))
            for i in range(len(shortest_path) - 1):
                graph[shortest_path[i]][shortest_path[i + 1]]['bandwidth'] -= bandwidth_demand
                link_flags[(shortest_path[i], shortest_path[i + 1])] = True

            custom_print(f"Successfully embedded link from VM{vm_source} to VM{vm_target} with path: {shortest_path}")

        else:
            custom_print(f"Failed to embed link from VM{vm_source} to VM{vm_target} due to insufficient bandwidth. Rolling back node and link embedding for VNR {vnr['vnr_id']}.")
            rollback_embedding(vm_to_server_assignments, path_mappings, servers, graph)
            return {}, graph, []

    return embedding_success, graph, path_mappings


def rollback_embedding(vm_to_server_assignments, path_mappings, servers, graph):
    # Rollback node embedding by releasing CPU resources
    for vm, server_id in vm_to_server_assignments.items():
        vm_cpu = next((v['cpu'] for v in servers[server_id]['vms'] if v['vnr_id'] == vm_to_server_assignments[vm]), 0)
        if vm_cpu > 0:
            servers[server_id]['cpu'] += vm_cpu
            servers[server_id]['vms'] = [v for v in servers[server_id]['vms'] if v['vnr_id'] != vm]
        custom_print(f"Rolled back {vm_cpu} CPU units for server {server_id}. New available CPU: {servers[server_id]['cpu']}")

    for (source_server, target_server, vnr_id), path in path_mappings:
        bandwidth_demand = 0
        for ((srv, tgt, vid), bandwidth_path) in path_mappings:
            if srv == source_server and tgt == target_server and vid == vnr_id and bandwidth_path == path:
                bandwidth_demand = bandwidth_path[1] if isinstance(bandwidth_path, tuple) and len(
                    bandwidth_path) > 1 else 0
                break

        if isinstance(bandwidth_demand, int) and bandwidth_demand > 0:
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                graph[node1][node2]['bandwidth'] += bandwidth_demand
                graph[node2][node1]['bandwidth'] += bandwidth_demand
                custom_print(f"Rolled back {bandwidth_demand} bandwidth on link {node1} <-> {node2}. New bandwidth: {graph[node1][node2]['bandwidth']}")

    custom_print("Rollback of node and link embedding completed.")

def main():
    with open(sys.argv[1], 'r') as f:
        vnr_info = json.load(f)

    with open(sys.argv[2], 'r') as f:
        SN_data = json.load(f)

    vnr_graph_counter = sys.argv[3]  # This is still a simple string argument

    with open(sys.argv[4], 'r') as f:
        vnr_graph = json.load(f)

    output_file_name = 'Node_Link_Embedding_Details.json'

    # Load substrate network data
    servers, graph, link_flags = load_sn_graph(SN_data)

    # Perform spectral clustering on substrate network
    clusters, croi, node_list = partition_substrate_network(SN_data, k1=5)
    weight_matrix = construct_weight_matrix(SN_data, node_list)

    vnr_graph_results = []

    for vnr in vnr_graph:
        vnr_id = vnr['vnr_id']
        custom_print(f"\nProcessing Node and Link Embeddings for VNR ID: {vnr_id}")

        # Perform Node Embedding
        vm_to_server_assignments, vnr_to_server_assignments, updated_servers = node_embedding_and_mapping(weight_matrix, vnr, SN_data, clusters, croi, node_list, graph, servers)

        if not vm_to_server_assignments:  # If node embedding fails
            custom_print(f"Node embedding failed for VNR {vnr_id}. Rolling back resources.")
            rollback_embedding(vm_to_server_assignments, [], servers, None)  # Rollback node embedding
            vnr_graph_results.append({
                "failure": True,
                "vnr_graph_num": vnr_graph_counter,
                "vnr_id": vnr_id,
                "details": "Node mapping failed. Check CPU/bandwidth requirements."
            })
            continue  # Skip this VNR as embedding failed

        custom_print(f"VM to Server Assignments for VNR ID {vnr_id}: {vm_to_server_assignments}")

        # Perform Link Embedding
        embedding_success, updated_graph, path_mappings = link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags, servers)

        if not embedding_success:  # If link embedding fails
            custom_print(f"Link embedding failed for VNR {vnr_id}. Rolling back node and link embedding.")
            rollback_embedding(vm_to_server_assignments, path_mappings, servers, graph)  # Rollback node and link embedding
            vnr_graph_results.append({
                "failure": True,
                "vnr_graph_num": vnr_graph_counter,
                "vnr_id": vnr_id,
                "details": "Link mapping failed. Check bandwidth requirements."
            })
            continue  # Skip this VNR as embedding failed

        custom_print(f"Path Mappings for VNR ID {vnr_id}: {path_mappings}")

        for node_id, node_data in SN_data.items():
            if isinstance(node_data, dict):
                updated_graph.add_node(node_id, **node_data)
                updated_graph.nodes[node_id].update(node_data)

        for key, value in updated_servers.items():
            updated_graph.nodes[key]['allocated_cores'] = value['cpu']
            updated_graph.nodes[key]['cpu'] = value['cpu']

        # Ensure JSON format matches Lite.py
        embedding_data = {
            "vnr_graph_num": vnr_graph_counter,
            "vnr_id": vnr_id,
            "vm_to_server_assignments": dict(vm_to_server_assignments),
            "path_mappings": path_mappings,
            "link_flags": list(link_flags.items()),
            "embedding_success": embedding_success,
            "updated_graph": nx.node_link_data(updated_graph)
        }
        vnr_graph_results.append(embedding_data)
    else:
        vnr_graph_results.append({
            "failure": True,
            "vnr_graph_num": vnr_graph_counter,
            "vnr_id": vnr_id,
            "details": "Node mapping failed. Check CPU/bandwidth requirements."
        })

    existing_data = []
    existing_data.append({
        "vnr_graph_num": vnr_graph_counter,
        "vnr_results": vnr_graph_results
    })

    with open(output_file_name, 'w') as file:
        json.dump(existing_data, file, indent=4)

    custom_print(f"\n Embedding results for VNR Graph {vnr_graph_counter} saved successfully.")

if __name__ == "__main__":
    main()