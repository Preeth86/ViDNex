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

        # Add nodes to graph
        for node_id, node_data in sn_json.items():
            if node_id.startswith('h'):
                servers[node_id] = {
                    'cpu': node_data.get('allocated_cores', 0),
                    'original_cpu': node_data.get('original_cores', 0),
                    'vms': []
                }
                graph.add_node(node_id, type='host', cpu=node_data.get('allocated_cores', 0))
            elif node_id.startswith('l') or node_id.startswith('s'):
                if isinstance(node_data, dict):
                    graph.add_node(node_id, type='switch', **node_data)
                else:
                    graph.add_node(node_id, type='switch')

        # Add edges (links between switches and servers)
        for link in sn_json.get('links_details', []):
            node1, node2 = link.get('node1'), link.get('node2')
            bandwidth = link.get('assigned_bandwidth', 0)
            graph.add_edge(node1, node2, bandwidth=bandwidth)

        # Ensure the controller connects to all spine switches
        graph.add_node('c0', type='controller')
        for node in graph.nodes:
            if node.startswith('s'):  # Connect to all spine switches
                graph.add_edge('c0', node, bandwidth=10000)  # High bandwidth for controller connections

        # Ensure all nodes from SN JSON are present
        for node_id in sn_json.keys():
            if node_id not in graph.nodes:
                graph.add_node(node_id)

        link_flags = {edge: False for edge in graph.edges()}

        visualize_graph(graph)  # Visualization for debugging
        return servers, graph, link_flags

    except KeyError as e:
        raise ValueError(f"Invalid SN JSON format. Missing key: {e}")
    except TypeError as e:
        raise ValueError(f"Type error while processing SN JSON: {e}")

def process_subnet(substrate_network_data):
    substrate_info = {}
    for sn, host_info in substrate_network_data.items():
        if not sn.startswith('h'):
            continue
        substrate_info[sn] = {'cpu': host_info.get('cpu', 0)}  # ✅ Correct key: 'cpu'
    return substrate_info

def create_clusters(vnr_info, data):
    custom_print("\nSubstrate Network CPU Available Details Before Clustering:")
    for node_name, info in data.items():
        if node_name.startswith('h'):
            custom_print(f"{node_name}: {info['cpu']}")  # FIXED: Use "cpu" instead of "allocated_cores"

    substrate_info = process_subnet(data)
    clusters = {}

    for vnr_name, vms_info in vnr_info.items():
        min_vm_cpu = min(vm_info['cpu'] for vm_info in vms_info.values())
        substrate_info_filtered = {sn: info for sn, info in substrate_info.items() if info['cpu'] > min_vm_cpu}

        if sum(len(vm) for vm in vms_info.values()) > len(substrate_info_filtered):
            custom_print(f"\nNumber of Virtual Machines in VNR {vnr_name} is greater than the number of available Substrate Nodes.")
            custom_print("Hence, mapping cannot happen.")
            return None

    for vnr_name, vms_info in vnr_info.items():
        clusters[vnr_name] = {vm_name: [] for vm_name in vms_info}
        for vm_name, vm_info in vms_info.items():
            for sn, sn_info in substrate_info.items():
                if sn_info['cpu'] >= vm_info['cpu']:  # FIXED: Use "cpu" instead of "allocated_cores"
                    clusters[vnr_name][vm_name].append(sn)

    custom_print("\nCluster Information:")
    for vnr_name, vnr_clusters in clusters.items():
        custom_print(f"{vnr_name}:")
        for vm_name, substrate_list in vnr_clusters.items():
            custom_print(f"  {vm_name}: {substrate_list}")
    return clusters

def solve_optimization(idx, clusters):
    pairs = [(vm_name, substrate_node) for vnr_name, vnr_clusters in clusters.items()
             for vm_name, substrate_list in vnr_clusters.items()
             for substrate_node in substrate_list]
    x = LpVariable.dicts('x', pairs, 0, 1, LpBinary)
    prob = LpProblem(f"VNR{idx}_Substrate_Mapping", LpMinimize)

    prob += lpSum(x[vm, substrate] for vm, substrate in pairs)

    for vnr_name, vnr_clusters in clusters.items():
        for vm_name, substrate_list in vnr_clusters.items():
            prob += lpSum(x[vm_name, substrate] for substrate in substrate_list) == 1

    for substrate in set(substrate_node for vnr_clusters in clusters.values()
                         for substrate_list in vnr_clusters.values()
                         for substrate_node in substrate_list):
        prob += lpSum(x[vm, substrate] for vnr_clusters in clusters.values()
                      for vm, substrate_list in vnr_clusters.items()
                      if substrate in substrate_list) <= 1

    pulp.LpSolverDefault.msg = False
    prob.solve()

    return x, prob

def node_embedding_and_mapping(servers, vnr):
    custom_print(f"\nNode Embedding and Mapping of VMs for VNR ID: {vnr['vnr_id']}")
    vm_to_server_assignments = {}
    vnr_to_server_assignments = {}

    # Step 1: Create Clusters
    clusters = create_clusters({
        f"VNR{vnr['vnr_id']}": {
            f"VM{vm + 1}": {"cpu": vnr['vm_cpu_cores'][vm]} for vm in range(len(vnr['vm_cpu_cores']))
        }
    }, servers)

    if clusters is None:
        custom_print(f"Failed to create clusters for VNR ID {vnr['vnr_id']}. Skipping embedding.")
        return {}, {}, servers  # Return empty mappings if clustering fails

    # Step 2: Solve Optimization Problem
    x, prob = solve_optimization(vnr['vnr_id'], clusters)

    # Step 3: Ensure Unique Servers Per VNR
    used_servers = set()

    for vm_name, substrate_list in clusters[f'VNR{vnr["vnr_id"]}'].items():
        mapped = False
        for substrate in substrate_list:
            if value(x[vm_name, substrate]) == 1 and substrate not in used_servers:
                servers[substrate]['cpu'] -= vnr['vm_cpu_cores'][int(vm_name[2:]) - 1]
                servers[substrate]['vms'].append({
                    'vnr_id': vnr['vnr_id'],
                    'vm_index': int(vm_name[2:]),
                    'cpu': vnr['vm_cpu_cores'][int(vm_name[2:]) - 1]
                })
                vm_to_server_assignments[vm_name] = substrate
                vnr_to_server_assignments.setdefault(vnr['vnr_id'], []).append(substrate)
                used_servers.add(substrate)  # Prevent another VM of the same VNR from using this server
                custom_print(f"{vm_name} mapped to {substrate} with {vnr['vm_cpu_cores'][int(vm_name[2:]) - 1]} CPU.")
                mapped = True
                break
        if not mapped:
            custom_print(f"\n{vm_name} failed to map to any Substrate Node.")
            return {}, {}, servers  # Return empty mappings if any VM fails

    custom_print(f"VM to Server Assignments for VNR ID {vnr['vnr_id']}: {vm_to_server_assignments}")
    return vm_to_server_assignments, vnr_to_server_assignments, servers

def k_shortest_paths(graph, start, end, k=1):
    paths = [(0, [start])]
    shortest_paths = []
    while paths and len(shortest_paths) < k:
        cost, path = heapq.heappop(paths)
        last_node = path[-1]
        if last_node == end:
            shortest_paths.append((cost, path))
            continue
        for next_node, edge_data in graph[last_node].items():
            if next_node not in path:
                heapq.heappush(paths, (cost + edge_data['bandwidth'], path + [next_node]))
    return shortest_paths

def link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags):
    custom_print(f"\nLink Embedding and Mapping of VLs for VNR ID: {vnr['vnr_id']}")
    embedding_success = {vnr['vnr_id']: True}
    path_mappings = []

    for link_index, (vm_source, vm_target) in enumerate(vnr['vm_links'], start=1):
        bandwidth_demand = vnr['bandwidth_values'][link_index - 1]

        if f"VM{vm_source}" not in vm_to_server_assignments or f"VM{vm_target}" not in vm_to_server_assignments:
            custom_print(f"Failed to find server assignments for VM{vm_source} or VM{vm_target}.")
            embedding_success[vnr['vnr_id']] = False
            break

        source_server = vm_to_server_assignments[f"VM{vm_source}"]
        target_server = vm_to_server_assignments[f"VM{vm_target}"]

        shortest_path = k_shortest_paths(graph, source_server, target_server, k=1)

        if shortest_path:
            selected_path = shortest_path[0][1]

            # Check if bandwidth is sufficient for all links in the selected path
            insufficient_bandwidth = any(
                graph[selected_path[i]][selected_path[i + 1]]['bandwidth'] < bandwidth_demand
                for i in range(len(selected_path) - 1)
            )

            if insufficient_bandwidth:
                custom_print(f"Failed to embed link from VM{vm_source} to VM{vm_target} due to insufficient bandwidth.")
                embedding_success[vnr['vnr_id']] = False
                break

            # Deduct bandwidth after successful validation
            path_mappings.append(((source_server, target_server, vnr['vnr_id']), selected_path))
            for i in range(len(selected_path) - 1):
                graph[selected_path[i]][selected_path[i + 1]]['bandwidth'] -= bandwidth_demand
                link_flags[(selected_path[i], selected_path[i + 1])] = True

            custom_print(f"Successfully embedded link from VM{vm_source} to VM{vm_target} with path: {selected_path}")

        else:
            custom_print(f"Failed to embed link from VM{vm_source} to VM{vm_target} due to insufficient bandwidth.")
            embedding_success[vnr['vnr_id']] = False
            break

    return embedding_success, graph, path_mappings


def main():
    with open(sys.argv[1], 'r') as f:
        vnr_info = json.load(f)

    with open(sys.argv[2], 'r') as f:
        SN_data = json.load(f)

    vnr_graph_counter = sys.argv[3]  # This is still a simple string argument

    with open(sys.argv[4], 'r') as f:
        vnr_graph = json.load(f)

    output_file_name = 'Node_Link_Embedding_Details.json'

    servers, graph, link_flags = load_sn_graph(SN_data)
    vnr_graph_results = []

    for vnr in vnr_graph:
        vnr_id = vnr['vnr_id']
        custom_print(f"\nProcessing Node and Link Embeddings for VNR ID: {vnr_id}")

        # Perform Node Embedding
        vm_to_server_assignments, vnr_to_server_assignments, updated_servers = node_embedding_and_mapping(servers, vnr)
        if vm_to_server_assignments:
            custom_print(f"VM to Server Assignments for VNR ID {vnr_id}: {vm_to_server_assignments}")

            # Perform Link Embedding
            embedding_success, updated_graph, path_mappings = link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags)

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
    custom_print(f"\nEmbedding results for VNR Graph {vnr_graph_counter} saved successfully.")

if __name__ == "__main__":
    main()