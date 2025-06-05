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

def calculate_mean_and_std(servers):
    cpu_resources = [server['cpu'] for server in servers.values()]  # Use current available CPU
    mean_cpu, std_cpu = np.mean(cpu_resources), np.std(cpu_resources, ddof=1)
    std_cpu = max(std_cpu, 1e-6)  # Prevent division by zero
    return mean_cpu, std_cpu

def calculate_link_bandwidth_statistics(graph):
    bandwidths = []
    for node in graph:
        for neighbor, bandwidth in graph[node].items():
            bandwidths.append(bandwidth)  # Append bandwidth of each link

    mean_bw_available = np.mean(bandwidths)
    std_bw_available = np.std(bandwidths, ddof=1)
    std_bw_available = max(std_bw_available, 1e-6)  # Prevent division by zero
    return mean_bw_available, std_bw_available

def node_embedding_and_mapping(servers, vnr):
    custom_print(f"\nNode Embedding and Mapping of VMs for VNR ID: {vnr['vnr_id']}")

    # Constraints: Power consumption constants and scaling factor
    P_idle, P_full, alpha_1 = 150, 300, 0.8  # Power constants and scaling factor for cost function

    vm_to_server_assignments = {}
    vnr_to_server_assignments = {}

    # Track used servers to prevent multiple VMs from the same VNR being mapped to the same server
    used_servers = set()

    # Calculate mean and std deviation of CPU resources
    mean_cpu, std_cpu = calculate_mean_and_std(servers)

    for vm_index, vm_cpu in enumerate(vnr['vm_cpu_cores'], start=1):
        best_server = None
        best_objective_value = float('inf')

        for server_id, server_info in servers.items():
            if server_info['cpu'] >= vm_cpu and server_id not in used_servers:  # Prevent multiple VMs on the same server
                U_cpu = (vm_cpu / server_info['cpu']) * 100
                overloading_prob = 1 - norm.cdf((server_info['cpu'] - vm_cpu - mean_cpu) / std_cpu)
                cumulative_cpu = vm_cpu + sum(v['cpu'] for v in server_info['vms'])
                P_k_U_cpu = P_idle + (P_full - P_idle) * (cumulative_cpu / server_info['original_cpu'])
                node_mapping_objective = P_k_U_cpu * exp(alpha_1 * overloading_prob)

                # Select server with the lowest node objective value
                if node_mapping_objective < best_objective_value:
                    best_objective_value = node_mapping_objective
                    best_server = server_id

        if best_server is None:
            custom_print(f"❌ No suitable server found for VM{vm_index}. Rolling back all node assignments for VNR {vnr['vnr_id']}.")
            rollback_embedding(vm_to_server_assignments, [], servers, None)  # Rollback node embedding
            return {}, {}, servers

        # Assign VM to the best server and update used_servers to prevent duplicate assignments
        servers[best_server]['cpu'] -= vm_cpu
        servers[best_server]['vms'].append({'vnr_id': vnr['vnr_id'], 'vm_index': vm_index, 'cpu': vm_cpu})
        vm_to_server_assignments[f"VM{vm_index}"] = best_server
        vnr_to_server_assignments.setdefault(vnr['vnr_id'], []).append(best_server)
        used_servers.add(best_server)  # Ensure no duplicate assignments for the same VNR

        custom_print(f"✅ VM{vm_index} mapped to {best_server} with {vm_cpu} CPU (Objective: {best_objective_value:.4f})")

    return vm_to_server_assignments, vnr_to_server_assignments, servers

def k_shortest_paths(graph, start, end, k=5):
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

def link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags, servers):
    custom_print(f"\nLink Embedding and Mapping of VLs for VNR ID: {vnr['vnr_id']}")
    embedding_success = {vnr['vnr_id']: True}
    path_mappings = []

    for link_index, (vm_source, vm_target) in enumerate(vnr['vm_links'], start=1):
        bandwidth_demand = vnr['bandwidth_values'][link_index - 1]

        if f"VM{vm_source}" not in vm_to_server_assignments or f"VM{vm_target}" not in vm_to_server_assignments:
            custom_print(f"❌ Failed to find server assignments for VM{vm_source} or VM{vm_target}. Rolling back node embedding for VNR {vnr['vnr_id']}.")
            rollback_embedding(vm_to_server_assignments, [], servers, None)  # Rollback node embedding
            return {}, graph, []

        source_server = vm_to_server_assignments[f"VM{vm_source}"]
        target_server = vm_to_server_assignments[f"VM{vm_target}"]

        shortest_path = k_shortest_paths(graph, source_server, target_server, k=5)

        if shortest_path:
            selected_path = shortest_path[0][1]

            # Check if bandwidth is sufficient for all links in the selected path
            insufficient_bandwidth = any(
                graph[selected_path[i]][selected_path[i + 1]]['bandwidth'] < bandwidth_demand
                for i in range(len(selected_path) - 1)
            )

            if insufficient_bandwidth:
                custom_print(f"❌ Insufficient bandwidth for embedding link from VM{vm_source} to VM{vm_target}. Rolling back node and link embedding for VNR {vnr['vnr_id']}.")
                rollback_embedding(vm_to_server_assignments, path_mappings, servers, graph)  # Rollback node and link embedding
                return {}, graph, []

            # Deduct bandwidth after successful validation
            path_mappings.append(((source_server, target_server, vnr['vnr_id']), selected_path))
            for i in range(len(selected_path) - 1):
                graph[selected_path[i]][selected_path[i + 1]]['bandwidth'] -= bandwidth_demand
                link_flags[(selected_path[i], selected_path[i + 1])] = True

            custom_print(f"✅ Successfully embedded link from VM{vm_source} to VM{vm_target} with path: {selected_path}")

        else:
            custom_print(f"❌ Failed to embed link from VM{vm_source} to VM{vm_target} due to insufficient bandwidth. Rolling back node and link embedding for VNR {vnr['vnr_id']}.")
            rollback_embedding(vm_to_server_assignments, path_mappings, servers, graph)  # Rollback node and link embedding
            return {}, graph, []

    return embedding_success, graph, path_mappings


def rollback_embedding(vm_to_server_assignments, path_mappings, servers, graph):
    # Rollback node embedding by releasing CPU resources
    for vm, server_id in vm_to_server_assignments.items():
        vm_cpu = next((v['cpu'] for v in servers[server_id]['vms'] if v['vnr_id'] == vm_to_server_assignments[vm]), 0)
        if vm_cpu > 0:
            servers[server_id]['cpu'] += vm_cpu
            servers[server_id]['vms'] = [v for v in servers[server_id]['vms'] if v['vnr_id'] != vm]
        custom_print(f"🔄 Rolled back {vm_cpu} CPU units for server {server_id}. New available CPU: {servers[server_id]['cpu']}")

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
                custom_print(f"🔄 Rolled back {bandwidth_demand} bandwidth on link {node1} <-> {node2}. New bandwidth: {graph[node1][node2]['bandwidth']}")

    custom_print("🔄 Rollback of node and link embedding completed.")

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

        if not vm_to_server_assignments:  # If node embedding fails
            custom_print(f"❌ Node embedding failed for VNR {vnr_id}. Rolling back resources.")
            rollback_embedding(vm_to_server_assignments, [], servers, None)  # Rollback node embedding
            vnr_graph_results.append({
                "failure": True,
                "vnr_graph_num": vnr_graph_counter,
                "vnr_id": vnr_id,
                "details": "Node mapping failed. Check CPU/bandwidth requirements."
            })
            continue  # Skip this VNR as embedding failed

        custom_print(f"✅ VM to Server Assignments for VNR ID {vnr_id}: {vm_to_server_assignments}")

        # Perform Link Embedding
        embedding_success, updated_graph, path_mappings = link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags, servers)

        if not embedding_success:  # If link embedding fails
            custom_print(f"❌ Link embedding failed for VNR {vnr_id}. Rolling back node and link embedding.")
            rollback_embedding(vm_to_server_assignments, path_mappings, servers, graph)  # Rollback node and link embedding
            vnr_graph_results.append({
                "failure": True,
                "vnr_graph_num": vnr_graph_counter,
                "vnr_id": vnr_id,
                "details": "Link mapping failed. Check bandwidth requirements."
            })
            continue  # Skip this VNR as embedding failed

        custom_print(f"✅ Path Mappings for VNR ID {vnr_id}: {path_mappings}")

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

    custom_print(f"\n✅ Embedding results for VNR Graph {vnr_graph_counter} saved successfully.")

if __name__ == "__main__":
    main()