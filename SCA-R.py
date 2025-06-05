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
from collections import defaultdict
from scipy.optimize import linprog
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

def SetPriority(VNR):
    high = []
    medium = []
    low = []
    for vnr in VNR:
        usp_value = vnr.get('usp_value', 0)  # Assuming usp_value is a property of vnr
        if usp_value >= 0.9999:
            high.append(vnr)
        elif usp_value >= 0.97:
            medium.append(vnr)
        else:
            low.append(vnr)
    return high, medium, low

def assign_usp_to_vnrs(VNR):
    for vnr in VNR:
        vnr_id = vnr['vnr_id']
        if vnr_id % 3 == 0:
            vnr['usp_value'] = 0.9999  # High priority
        elif vnr_id % 3 == 1:
            vnr['usp_value'] = 0.97  # Medium priority
        else:
            vnr['usp_value'] = 0.90  # Low priority
    return VNR

# DiVINE Node and Link Mapping
def create_augmented_graph(SN_data, vnr):
    augmented_graph = defaultdict(dict)  # No more NameError
    meta_nodes = {}

    for vm_index, vm_cpu in enumerate(vnr['vm_cpu_cores']):
        meta_node = f"VM{vm_index + 1}"
        meta_nodes[meta_node] = []

        for server in SN_data:
            if isinstance(SN_data[server], dict) and SN_data[server].get('cpu', 0) >= vm_cpu:
                augmented_graph[meta_node][server] = 0  # Infinite capacity for meta edges
                meta_nodes[meta_node].append(server)

    if 'links_details' in SN_data:
        for link in SN_data['links_details']:
            u, v = link['node1'], link['node2']
            augmented_graph[u][v] = link['assigned_bandwidth']
            augmented_graph[v][u] = link['assigned_bandwidth']

    return augmented_graph, meta_nodes

def solve_lp(augmented_graph, meta_nodes, vnr):
    c = []
    A_eq = []
    b_eq = []
    bounds = []

    server_list = []
    for meta_node, servers in meta_nodes.items():
        server_list.extend(servers)
        for server in servers:
            c.append(1)

    server_count = len(server_list)

    # Constraint: Each VM must be mapped to exactly one server
    for meta_node, servers in meta_nodes.items():
        constraint = [0] * server_count
        for server in servers:
            idx = server_list.index(server)
            constraint[idx] = 1
        A_eq.append(constraint)
        b_eq.append(1)

    for _ in c:
        bounds.append((0, 1))

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        return res.x, server_list
    else:
        custom_print(f"❌ LP Solver failed: {res.message}")
        return None, None

def deterministic_rounding(lp_solution, meta_nodes, server_list):
    vm_to_server_assignments = {}
    for meta_node, servers in meta_nodes.items():
        best_server = None
        best_value = 0
        for server in servers:
            idx = server_list.index(server)
            if lp_solution[idx] > best_value:
                best_value = lp_solution[idx]
                best_server = server
        if best_server is not None:
            vm_to_server_assignments[meta_node] = best_server
    return vm_to_server_assignments

# SCA-R Optimization Functions with USP
def traffic_statistics_collection(servers, time_slot=10):
    traffic_stats = {server: sum(random.randint(0, 100) for _ in range(time_slot)) for server in servers}
    return traffic_stats

def usp_constraint(value, mu, sigma, Pg):
    part1 = mu * (1 - Pg)
    part2 = sigma * np.sqrt(2) * erf(Pg)
    return part1 + part2

def extract_connections(path_mappings, bandwidth_values):
    connections = []
    for link_index, path in path_mappings.items():
        bandwidth = bandwidth_values[link_index - 1]
        connections.append((link_index, path, bandwidth))
    return connections

def sca_r_optimization(priority_vnrs, traffic_stats, servers, graph, all_embedding_results, output_file_name):
    server_keys = list(servers.keys())
    server_indices = list(range(len(server_keys)))

    for vnr in priority_vnrs:
        custom_print(f"\nRe-embedding using SCA-R for VNR ID: {vnr['vnr_id']+1}")

        # Re-embedding logic starts here with updated constraints
        vm_to_server_assignments, servers = node_embedding_and_mapping(servers, vnr, avoid_same_server=True)
        custom_print(f"Re-embedded VM to Server Assignments: {vm_to_server_assignments}")

        embedding_success, graph, path_mappings = link_embedding_and_mapping(graph, vnr, vm_to_server_assignments)
        custom_print(f"Re-embedded Path Mappings: {path_mappings}")

        all_embedding_results.append((vnr, embedding_success, vm_to_server_assignments, path_mappings))

        # Optimization constraints to avoid mapping multiple VMs of the same VNR to the same server
        def objective(x):
            x = np.round(x).astype(int)
            server_ids = [server_keys[idx] for idx in x if 0 <= idx < len(server_keys)]
            return sum(traffic_stats[server_id] for server_id in server_ids)

        constraints = [{'type': 'ineq',
                        'fun': lambda x: servers[server_keys[int(idx)]]['cpu'] - traffic_stats[server_keys[int(idx)]]}
                       for idx in range(len(servers))]

        # Add constraint to avoid mapping multiple VMs of the same VNR to the same server
        def vm_separation_constraint(x):
            x = np.round(x).astype(int)
            server_counts = defaultdict(int)
            for idx in x:
                if 0 <= idx < len(server_keys):
                    server_counts[server_keys[idx]] += 1
            # Ensure no server has more than one VM from the same VNR
            return min(1 - count for count in server_counts.values())

        constraints.append({'type': 'ineq', 'fun': vm_separation_constraint})

        usp_constraints = [{'type': 'ineq',
                            'fun': lambda x, mu=traffic_stats[server_keys[int(idx)]], sigma=10, Pg=0.95: usp_constraint(
                                x[int(idx)], mu, sigma, Pg)} for idx in range(len(servers))]

        bounds = [(0, len(server_keys) - 1) for _ in server_indices]

        result = minimize(objective, server_indices, constraints=constraints + usp_constraints, bounds=bounds)

        if result.success:
            optimized_indices = np.round(result.x).astype(int)
            optimized_servers = [server_keys[idx] for idx in optimized_indices if 0 <= idx < len(server_keys)]
        #     custom_print(f"Optimized Server Assignments: {optimized_servers}")
        # else:
        #     custom_print(f"Optimization failed for VNR ID: {vnr['vnr_id']}")

        def relaxed_objective(flow_vars):
            penalty = sum(2 * (z - z ** 2) for z in flow_vars)
            return np.sum(flow_vars) + penalty

        bounds = [(0, 1) for _ in graph]
        flow_vars_initial_guess = [0.5 for _ in graph]

        relaxed_constraints = [{'type': 'ineq',
                                'fun': lambda x, idx=idx: servers[server_keys[idx]]['cpu'] - traffic_stats[
                                    server_keys[idx]] * x[idx]} for idx in range(len(servers))]
        relaxed_result = minimize(relaxed_objective, flow_vars_initial_guess, bounds=bounds,
                                  constraints=relaxed_constraints)

        if relaxed_result.success:
            relaxed_flow_vars = relaxed_result.x
        #     custom_print(f"Relaxed Flow Variables: {relaxed_flow_vars}")
        # else:
        #     custom_print(f"Relaxation failed for VNR ID: {vnr['vnr_id']}")

        # Save the final re-embedding results to the pickle file
        embedding_data = [list(vm_to_server_assignments.items()), list(path_mappings.items()), True, graph]
        with open(output_file_name, 'wb') as file:
            pickle.dump(embedding_data, file)


        # Verify the contents of the pickle file
        custom_print(f"Saved embedding data: {embedding_data}")
        return extract_connections(path_mappings, vnr['bandwidth_values'])

def node_embedding_and_mapping(SN_data, vnr, avoid_same_server=True):
    vnr_id = vnr['vnr_id']
    custom_print(f"\nStarting node embedding for VNR ID: {vnr_id}")

    # Step 1: Create Augmented Graph for LP-based Optimization
    augmented_graph, meta_nodes = create_augmented_graph(SN_data, vnr)

    # Step 2: Solve LP Optimization for Initial Placement
    lp_solution, server_list = solve_lp(augmented_graph, meta_nodes, vnr)

    if lp_solution is None or server_list is None:
        custom_print(f"❌ LP Solver failed for VNR ID: {vnr_id}. No valid solution found.")
        return {}, {}, SN_data  # Return empty assignments if LP fails

    custom_print(f"Server List: {server_list}")

    # Step 3: Perform Deterministic Rounding to Assign Servers
    vm_to_server_assignments = deterministic_rounding(lp_solution, meta_nodes, server_list)
    custom_print(f"Initial VM to Server Assignments: {vm_to_server_assignments}")

    if avoid_same_server:
        # Ensure no two VMs of the same VNR are mapped to the same server
        used_servers = set()
        for vm, server in list(vm_to_server_assignments.items()):
            if server in used_servers:
                # Try to find an alternative server that meets the VM's CPU requirement
                for alt_server in SN_data:
                    if alt_server not in used_servers and SN_data[alt_server]['cpu'] >= vnr['vm_cpu_cores'][
                        int(vm.split('VM')[1]) - 1]:
                        vm_to_server_assignments[vm] = alt_server
                        used_servers.add(alt_server)
                        break
            else:
                used_servers.add(server)

    custom_print(f"Final VM to Server Assignments after enforcing unique servers per VNR: {vm_to_server_assignments}")

    # Step 4: Update Substrate Network Resources
    for meta_node, server in vm_to_server_assignments.items():
        if server is None:
            continue  # Skip if no valid server is assigned

        vm_index = int(meta_node.split('VM')[1]) - 1
        SN_data[server]['cpu'] -= vnr['vm_cpu_cores'][vm_index]
        SN_data[server]['vms'].append(
            {'vnr_id': vnr_id, 'vm_index': vm_index, 'cpu': vnr['vm_cpu_cores'][vm_index]}
        )

    # Ensure it returns three values (Added vnr_to_server_assignments)
    vnr_to_server_assignments = {vnr_id: list(vm_to_server_assignments.values())}

    return vm_to_server_assignments, vnr_to_server_assignments, SN_data  # ✅ Now returns 3 values

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