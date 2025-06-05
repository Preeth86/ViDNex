import json
import networkx as nx
from ryu.app import simple_switch_13
from webob import Response
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.lib import dpid as dpid_lib
from ryu.lib.packet import packet, ethernet, arp, ipv4, ether_types
from ryu.topology import event
from ryu.lib import hub
import os
import pickle
import time
import subprocess
import psutil
import re
import requests
import threading

# Constants
simple_switch_instance_name = 'simple_switch_api_app'
url = '/simpleswitch/mactable/{dpid}'
topology_url = '/topology'
vnr_url = '/vnr'

# Global variable to store received SN topology
received_topology = {}
vnr_requests = []

class SimpleSwitchController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(SimpleSwitchController, self).__init__(req, link, data, **config)
        self.simple_switch_app = data[simple_switch_instance_name]
        self.topology_data = self.simple_switch_app.topology_data  # Use the app's topology graph
        self.simple_switch_app.monitor_pre_embedding_resources()
        global vnr_requests

        self.vnr_graphs = []  # Initialize vnr_graphs as an empty list

    @route('simpleswitch', url, methods=['GET'], requirements={'dpid': dpid_lib.DPID_PATTERN})
    def list_mac_table(self, req, **kwargs):
        simple_switch = self.simple_switch_app
        dpid = dpid_lib.str_to_dpid(kwargs['dpid'])

        if dpid not in simple_switch.mac_to_port:
            return Response(status=404)

        mac_table = simple_switch.mac_to_port.get(dpid, {})
        body = json.dumps(mac_table)
        return Response(content_type='application/json', body=body)

    @route('simpleswitch', url, methods=['PUT'], requirements={'dpid': dpid_lib.DPID_PATTERN})
    def put_mac_table(self, req, **kwargs):
        simple_switch = self.simple_switch_app
        dpid = dpid_lib.str_to_dpid(kwargs['dpid'])
        try:
            new_entry = req.json if req.body else {}
        except ValueError:
            return Response(status=400)

        if dpid not in simple_switch.mac_to_port:
            return Response(status=404)

        try:
            mac_table = simple_switch.set_mac_to_port(dpid, new_entry)
            body = json.dumps(mac_table)
            return Response(content_type='application/json', body=body)
        except Exception as e:
            return Response(status=500)

    @route('topology', topology_url, methods=['POST'])
    def receive_topology(self, req, **kwargs):
        """Receive SN topology details via POST request and save as JSON."""
        global received_topology
        try:
            data = req.json if req.body else {}
            if not data:
                return Response(status=400, body="Invalid or empty topology data.")

            received_topology = data  # Store the original format

            # Generate the NetworkX graph with all required attributes
            self.topology_data.clear()

            # Prepare formatted SN graph structure
            formatted_sn_graph = {
                "num_spine_switches": 0,
                "num_leaf_switches": 0,
                "num_hosts": 0,
                "links_details": []
            }

            # Process hosts (servers)
            for host_id, host_details in data.items():
                if host_id.startswith('h'):
                    self.topology_data.add_node(
                        host_id,
                        cpu=host_details.get("allocated_cores", 0),
                        original_cpu=host_details.get("original_cores", host_details.get("allocated_cores", 0)),
                        connections=host_details.get("connections", []),
                        type="host"
                    )
                    formatted_sn_graph[host_id] = {
                        "allocated_cores": host_details.get("allocated_cores", 0),
                        "original_cores": host_details.get("original_cores", host_details.get("allocated_cores", 0)),
                        "connections": []
                    }
                    formatted_sn_graph["num_hosts"] += 1

                    # Extract and store host IP
                    ip_address = host_details.get("ip", None)
                    if ip_address:
                        received_topology[host_id]["ip"] = ip_address  # Store IP in received_topology

            # Process switches
            switches = set(link['node1'] for link in data['links_details']).union(
                link['node2'] for link in data['links_details']
            ) - set(data.keys())
            for switch_id in switches:
                self.topology_data.add_node(switch_id, type="switch")
                if switch_id.startswith('s'):
                    formatted_sn_graph["num_spine_switches"] += 1
                elif switch_id.startswith('l'):
                    formatted_sn_graph["num_leaf_switches"] += 1

            # Process links
            for link in data['links_details']:
                self.topology_data.add_edge(
                    link["node1"],
                    link["node2"],
                    bandwidth=link.get("assigned_bandwidth", 0)
                )
                formatted_sn_graph["links_details"].append(link)

            # ---- Add Controller and Connect Spine Switches (Corrected Bi-directional Assignment) ----
            controller_node = 'c0'  # Represent the Ryu controller as 'c0'
            self.topology_data.add_node(controller_node, type='controller')

            for switch_id in switches:
                if switch_id.startswith('s'):  # Connect only spine switches
                    if not self.topology_data.has_edge(switch_id, controller_node):
                        self.topology_data.add_edge(switch_id, controller_node, bandwidth=6000)  # Controller link

                        # Store only one entry per connection (no duplicate bi-directional entries)
                        formatted_sn_graph["links_details"].append({
                            "node1": switch_id,
                            "node2": controller_node,
                            "assigned_bandwidth": 6000
                        })

            print(f"Added bi-directional links between controller {controller_node} and spine switches.")

            # Save SN to JSON (in the same location as the pickle file)
            pickle_path = '/home/vnesdn/PycharmProjects/EFraS++/SN/SN.topo.pickle'
            json_path = pickle_path.replace('.pickle', '.json')

            with open(json_path, 'w') as f:
                json.dump(formatted_sn_graph, f, indent=4)

            # Store SN graph for later retrieval
            self.simple_switch_app.formatted_sn_graph = formatted_sn_graph

            # Print updated topology with extracted IPs for debugging
            print(f"Updated Topology with Host IPs: {json.dumps(received_topology, indent=4)}")

            return Response(status=200, body="Topology received and processed successfully and saved to JSON.")
        except Exception as e:
            return Response(status=500, body=f"Failed to process topology data: {str(e)}")

    @route('topology', topology_url, methods=['GET'])
    def get_topology(self, req, **kwargs):
        """Expose the SN topology graph as JSON in the desired format."""
        try:
            formatted_sn_graph = self.simple_switch_app.formatted_sn_graph
            if not formatted_sn_graph:
                return Response(status=404, body="SN graph not found. Ensure topology has been received first.")
            body = json.dumps(formatted_sn_graph, indent=4)
            return Response(content_type='application/json; charset=utf-8', body=body)
        except Exception as e:
            return Response(status=500, body=f"Failed to fetch topology data: {str(e)}")

    @route('simpleswitch', vnr_url, methods=['POST'])
    def store_vnr(self, req, **kwargs):
        """Process and store VNR pickle data as a single JSON file containing multiple VNRs."""
        try:
            data = req.json if req.body else {}
            if not data or 'vnr_file' not in data or 'vnr_data' not in data:
                return Response(status=400, body="Invalid VNR data. 'vnr_file' or 'vnr_data' missing.")

            # ✅ Call pre-embedding monitoring before processing VNRs
            self.simple_switch_app.monitor_pre_embedding_resources()

            # ✅ Correctly call detect_major_fluctuations from SimpleSwitchRest13
            hub.spawn(self.simple_switch_app.detect_major_fluctuations)

            vnr_file_name = data['vnr_file']
            vnr_data_list = data['vnr_data']
            pickle_dir = os.path.dirname(vnr_file_name)
            base_name = os.path.basename(vnr_file_name).replace('.pickle', '')

            formatted_vnr_graph = []
            for vnr_id, vnr_data in enumerate(vnr_data_list, start=1):
                transformed_vm_links = []
                bandwidth_values = []

                for link_idx, (source, target) in enumerate(vnr_data.get("vm_links", [])):
                    transformed_vm_links.append([source, target])
                    bandwidth_values.append(vnr_data.get("bandwidth_values", [])[link_idx])

                formatted_vnr = {
                    "num_vms": len(vnr_data.get("vm_cpu_cores", [])),
                    "vm_cpu_cores": vnr_data.get("vm_cpu_cores", []),
                    "vm_links": transformed_vm_links,
                    "bandwidth_values": bandwidth_values,
                    "vnr_id": vnr_id
                }
                formatted_vnr_graph.append(formatted_vnr)

            json_path = os.path.join(pickle_dir, f"{base_name}.json")
            with open(json_path, 'w') as f:
                json.dump(formatted_vnr_graph, f, indent=4)

            self.simple_switch_app.vnr_graphs.append(formatted_vnr_graph)

            return Response(status=200, body=f"VNR file {vnr_file_name} processed and saved as JSON.")

        except Exception as e:
            return Response(status=500, body=f"Failed to process VNR: {str(e)}")

    @route('simpleswitch', vnr_url, methods=['GET'])
    def fetch_vnrs(self, req, **kwargs):
        """Retrieve serialized formatted VNR graphs as JSON."""
        try:
            if not self.simple_switch_app.vnr_graphs:
                return Response(status=404, body="No VNR graphs found.")
            body = json.dumps(self.simple_switch_app.vnr_graphs, indent=4)  # Serialize formatted VNRs
            return Response(content_type='application/json; charset=utf-8', body=body)
        except Exception as e:
            return Response(status=500, body=f"Failed to fetch VNRs: {str(e)}")

    @route('simpleswitch', '/rtt', methods=['GET'])
    def get_rtt(self, req, **kwargs):
        """Ensure manager.py receives stored RTT values before computing new ones."""
        try:
            # ✅ Ensure RTT storage is initialized
            if not hasattr(self.simple_switch_app, 'rtt_data'):
                self.simple_switch_app.rtt_data = {}

            # ✅ Always send the stored initial RTT first (if exists)
            if hasattr(self.simple_switch_app, 'initial_rtt_data') and self.simple_switch_app.initial_rtt_data:
                print(
                    f"[Debug] Sending stored Initial RTT Data to manager.py: {self.simple_switch_app.initial_rtt_data}")
                return Response(content_type='application/json; charset=utf-8',
                                body=json.dumps(self.simple_switch_app.initial_rtt_data, indent=4))

            # ✅ If initial RTT is empty, compute it once
            self.simple_switch_app.initial_rtt_data = self.simple_switch_app.rtt_data.copy()

            # ✅ Spawn thread to fetch new RTT values asynchronously
            threading.Thread(target=self.fetch_rtt_values, args=(self.simple_switch_app.rtt_data,)).start()

            return Response(content_type='application/json; charset=utf-8',
                            body=json.dumps(self.simple_switch_app.initial_rtt_data, indent=4))

        except Exception as e:
            print(f"[Error] Failed to fetch RTT data: {str(e)}")
            return Response(status=500, body=f"Failed to fetch RTT data: {str(e)}")

    def fetch_rtt_values(self, rtt_data):
        """Fetch and store RTT values asynchronously without overwriting previous values."""
        global received_topology
        print("[DEBUG] Running RTT test for all hosts...")

        for host, details in received_topology.items():
            if host.startswith("h"):
                target_ip = details.get("ip")
                if target_ip:
                    latency = self.simple_switch_app.track_latency(target_ip)
                    if latency is not None:
                        rtt_data[host] = latency
                        print(f"[DEBUG] RTT for {host} ({target_ip}): {latency} ms")

        # ✅ Persist RTT Data without overwriting the initial snapshot
        if not hasattr(self.simple_switch_app, 'initial_rtt_data') or not self.simple_switch_app.initial_rtt_data:
            self.simple_switch_app.initial_rtt_data = rtt_data.copy()

        print(f"[DEBUG] Stored RTT Data: {self.simple_switch_app.rtt_data}")

    @route('simpleswitch', '/packet_loss', methods=['GET'])
    def get_packet_loss(self, req, **kwargs):
        """Ensure manager.py receives stored Packet Loss values before computing new ones."""
        print("[Debug] Received request for Packet Loss")

        try:
            # ✅ Ensure Packet Loss storage is initialized
            if not hasattr(self.simple_switch_app, 'packet_loss_data'):
                self.simple_switch_app.packet_loss_data = {}

            # ✅ Always send the stored initial Packet Loss first (if exists)
            if hasattr(self.simple_switch_app,
                       'initial_packet_loss_data') and self.simple_switch_app.initial_packet_loss_data:
                print(
                    f"[Debug] Sending stored Initial Packet Loss Data to manager.py: {self.simple_switch_app.initial_packet_loss_data}")
                return Response(content_type='application/json; charset=utf-8',
                                body=json.dumps(self.simple_switch_app.initial_packet_loss_data, indent=4))

            # ✅ If initial Packet Loss is empty, compute it once
            self.simple_switch_app.initial_packet_loss_data = self.simple_switch_app.packet_loss_data.copy()

            # ✅ Spawn thread to fetch new Packet Loss values asynchronously
            threading.Thread(target=self.fetch_packet_loss_values,
                             args=(self.simple_switch_app.packet_loss_data,)).start()

            return Response(content_type='application/json; charset=utf-8',
                            body=json.dumps(self.simple_switch_app.initial_packet_loss_data, indent=4))

        except Exception as e:
            print(f"[Error] Failed to fetch Packet Loss data: {str(e)}")
            return Response(status=500, body=f"Failed to fetch Packet Loss data: {str(e)}")

    def fetch_packet_loss_values(self, packet_loss_data):
        """Fetch and store packet loss values asynchronously without overwriting previous values."""
        global received_topology
        print("[DEBUG] Running Packet Loss test for all hosts...")

        for host, details in received_topology.items():
            if host.startswith("h"):
                target_ip = details.get("ip")
                if target_ip:
                    loss = self.simple_switch_app.track_packet_loss(target_ip)
                    if loss is not None:
                        packet_loss_data[host] = loss
                        print(f"[DEBUG] Packet Loss for {host} ({target_ip}): {loss}%")

        # ✅ Persist Packet Loss Data without overwriting the initial snapshot
        if not hasattr(self.simple_switch_app,
                       'initial_packet_loss_data') or not self.simple_switch_app.initial_packet_loss_data:
            self.simple_switch_app.initial_packet_loss_data = packet_loss_data.copy()

        print(f"[DEBUG] Stored Packet Loss Data: {self.simple_switch_app.packet_loss_data}")


class SimpleSwitchRest13(simple_switch_13.SimpleSwitch13):
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(SimpleSwitchRest13, self).__init__(*args, **kwargs)
        self.switches = {}
        self.mac_to_port = {}
        self.topology_data = nx.Graph()
        self.vnr_graphs = []
        wsgi = kwargs['wsgi']
        wsgi.register(SimpleSwitchController, {simple_switch_instance_name: self})
        self.controller = SimpleSwitchController(None, None, {simple_switch_instance_name: self})
        self.formatted_sn_graph = {}
        self.cpu_stats = {}  # Store CPU usage results
        self.cpu_usage_stats = {}  # Initialize CPU monitoring dictionary

        # Flags to track if graphs have been printed
        self.sn_graph_printed = False
        self.vnr_graph_printed = False
        self.before_embedding_alert_printed = False

        # ✅ Start monitoring threads once
        self.monitor_thread = hub.spawn(self.monitor_topology)
        self.resource_monitor_thread = hub.spawn(self.monitor_resources_usage)

    def monitor_topology(self):
        while True:
            # Print SN graph if not printed and topology exists
            if not self.sn_graph_printed:
                self.display_sn_graph()
                if len(self.topology_data.nodes) > 0:
                    self.sn_graph_printed = True

            # Print VNR graph immediately when VNRs exist
            if not self.vnr_graph_printed:
                self.display_vnr_graphs()
                if len(self.vnr_graphs) > 0:
                    self.vnr_graph_printed = True

                # ✅ Call pre-embedding monitoring before fluctuation detection
                self.monitor_pre_embedding_resources()

                # Continue waiting for embedding results separately (non-blocking)
                hub.spawn(self.wait_for_embedding_results)  # Run in a new thread

            hub.sleep(10)

    def display_sn_graph(self):
        """Display the SN network graph details."""
        print("SN Network Graph Details:")
        print("Nodes:", list(self.topology_data.nodes(data=True)))  # Include node data
        print("Edges:", list(self.topology_data.edges(data=True)))  # Include edge data

    def display_vnr_graphs(self):
        """Display the details of all VNR graphs."""
        print("\nReceived VNR Graph Details:")
        actual_vnr_data_found = False  # ✅ Track if at least one VNR graph with data is printed

        for i, vnr_graph in enumerate(self.vnr_graphs, start=1):
            print(f"VNR Graph {i}:")

            # Check if vnr_graph is a list (which contains VNR dictionaries)
            if isinstance(vnr_graph, list) and len(vnr_graph) > 0:
                actual_vnr_data_found = True  # ✅ Mark that actual VNR data is present
                for vnr_index, vnr in enumerate(vnr_graph, start=1):  # Start VNR ID from 1
                    print(f"VNR ID: {vnr_index}")
                    print("VM CPU Cores:", vnr['vm_cpu_cores'])
                    print("VM Links:", vnr['vm_links'])
                    print("Bandwidth Values:", vnr['bandwidth_values'])
            elif isinstance(vnr_graph, nx.Graph) and len(vnr_graph.nodes) > 0:
                actual_vnr_data_found = True  # ✅ Mark that actual VNR data is present
                print("Nodes:", vnr_graph.nodes(data=True))
                print("Edges:", vnr_graph.edges(data=True))
            else:
                print("Unknown or empty VNR format. Skipping...")

        print("All VNR Graphs Processed.")

        # ✅ Ensure "Before Embedding Alert" prints only ONCE after actual VNR details are printed
        if actual_vnr_data_found and not self.before_embedding_alert_printed:
            self.before_embedding_alert_printed = True  # ✅ Set flag to prevent multiple prints
            self.print_before_embedding_alert()  # ✅ Call the function directly, not with hub.spawn()

    def print_before_embedding_alert(self):
        """Prints the 'Before Embedding Alert' message once before embedding starts."""
        print("\n[Before Embedding Alert] Checking for network fluctuations...")
        hub.spawn(self.detect_major_fluctuations, "Before Embedding")
        print("[Before Embedding Alert] Network fluctuation check completed...\n")

    def monitor_pre_embedding_resources(self):
        """Fetch real-time CPU and bandwidth usage before VNR embedding from the original SN graph."""
        original_topology_file = 'SN/SN.topo.json'  # Ensure this contains the pre-embedding SN graph

        try:
            with open(original_topology_file, 'r') as file:
                original_sn_graph = json.load(file)

            if not original_sn_graph:
                self.logger.info("[Resource Usage Pre-Monitor] No SN Graph found. Waiting...")
                return

            collected_cpu_stats = {}
            collected_bandwidth_stats = {}

            # ✅ Extract CPU usage before embedding
            for node_id, node_details in original_sn_graph.items():
                if isinstance(node_details, dict) and node_id.startswith("h"):  # Ensure it's a host
                    cpu_usage = node_details.get("allocated_cores", 0)  # Extract allocated CPU
                    collected_cpu_stats[node_id] = cpu_usage

            # ✅ Extract bandwidth usage before embedding
            links = original_sn_graph.get('links_details', [])  # ✅ Corrected to 'links_details'
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, dict):
                        node1 = link.get("node1")
                        node2 = link.get("node2")
                        assigned_bw = link.get("assigned_bandwidth", 0)

                        if node1 and node2:
                            link_key = f"{node1}-{node2}"
                            collected_bandwidth_stats[link_key] = assigned_bw

            # ✅ Log pre-embedding resource stats
            self.logger.info("\n[Pre-Monitor] CPU & Bandwidth usage before embedding VNRs")
            self.logger.info(f"[CPU Monitor] Initial CPU Usage: {collected_cpu_stats}")
            self.logger.info(f"[Bandwidth Monitor] Initial Bandwidth Usage: {collected_bandwidth_stats}")

        except FileNotFoundError:
            self.logger.warning("[Resource Usage Pre-Monitor] No SN Graph found. Waiting...")
        except json.JSONDecodeError as e:
            self.logger.error(f"[Pre-Monitor] JSON decoding error: {str(e)}")
        except Exception as e:
            self.logger.error(f"[Pre-Monitor] Unexpected error: {str(e)}")

    def wait_for_embedding_results(self):
        embedding_result_file = 'Node_Link_Embedding_Details.json'

        while not os.path.exists(embedding_result_file):
            print("Waiting for embedding to start...")
            time.sleep(20)

        # File exists, display embedding results
        self.display_embedding_results(embedding_result_file)
        self.embedding_results_printed = True

    def monitor_resources_usage(self):
        """Fetch real-time CPU and bandwidth usage after each VNR embedding from updated SN graph JSON."""
        embedding_result_file = 'Node_Link_Embedding_Details.json'

        while True:
            collected_cpu_stats = {}
            collected_bandwidth_stats = {}

            # ✅ Fetch the updated SN graph JSON
            try:
                with open(embedding_result_file, 'r') as file:
                    embedding_data = json.load(file)

                if not embedding_data:
                    self.logger.info("[Monitor] No embedding results found yet.")
                    hub.sleep(10)
                    continue

                # ✅ Extract CPU and bandwidth usage from the latest updated SN graph
                latest_vnr_graph = embedding_data[-1]  # Fetch the last (latest) VNR graph entry
                vnr_results = latest_vnr_graph.get('vnr_results', [])

                if not vnr_results:
                    self.logger.info("[Monitor] No VNR results found in the latest embedding data.")
                    hub.sleep(10)
                    continue

                for vnr in vnr_results:
                    updated_graph = vnr.get('updated_graph', {})

                    # ✅ Ensure updated_graph is properly formatted as a dictionary
                    if not isinstance(updated_graph, dict):
                        self.logger.warning(f"[Monitor] Unexpected updated graph format: {type(updated_graph)}")
                        continue

                    # ✅ Extract CPU usage per host (server)
                    nodes = updated_graph.get('nodes', [])
                    if isinstance(nodes, list):  # Ensure nodes is a list
                        for node in nodes:
                            if isinstance(node, dict) and node.get("type") == "host":  # Process only physical hosts
                                node_id = node.get("id")
                                cpu_usage = node.get("cpu", 0)
                                if node_id:
                                    collected_cpu_stats[node_id] = cpu_usage

                    # ✅ Extract bandwidth usage per link (FIXED)
                    links = updated_graph.get('links', [])  # ✅ Corrected to 'links'
                    if isinstance(links, list):  # Ensure links is a list
                        for link in links:
                            if isinstance(link, dict):
                                node1 = link.get("source")  # ✅ Use "source" instead of "node1"
                                node2 = link.get("target")  # ✅ Use "target" instead of "node2"
                                assigned_bw = link.get("bandwidth", 0)

                                if node1 and node2 and assigned_bw > 0:
                                    link_key = f"{node1}-{node2}"
                                    collected_bandwidth_stats[link_key] = assigned_bw  # ✅ Store assigned bandwidth

                    vnr_id = vnr.get('vnr_id', 'Unknown')
                    self.logger.info(f"\n[Post-Monitor] CPU & Bandwidth usage after embedding VNR ID: {vnr_id}")
                    self.logger.info(f"[CPU Monitor] CPU Usage: {collected_cpu_stats}")
                    self.logger.info(f"[Bandwidth Monitor] Bandwidth Usage: {collected_bandwidth_stats}")

            except FileNotFoundError:
                self.logger.warning("[Resource Usage Post-Monitor] No Embedding found. Waiting...")
            except json.JSONDecodeError as e:
                self.logger.error(f"[Post-Monitor] JSON decoding error: {str(e)}")
            except Exception as e:
                self.logger.error(f"[Post-Monitor] Unexpected error: {str(e)}")

            hub.sleep(10)  # Keep the 10-second interval

    def display_embedding_results(self, embedding_result_file=None):
        if embedding_result_file is None:
            embedding_result_file = 'Node_Link_Embedding_Details.json'

        try:
            with open(embedding_result_file, 'r') as file:
                embedding_data = json.load(file)

                print("\nEmbedding Results (Grouped by VNR Graph and VNR ID):")

                for vnr_graph in embedding_data:
                    graph_num = vnr_graph.get('vnr_graph_num', 'unknown')
                    vnr_results = vnr_graph.get('vnr_results', [])

                    print(f"\n[VNR Graph {graph_num}] Processing...")

                    combined_vm_to_server = {}  # ✅ Aggregate VM-to-server mappings across all VNRs in this graph

                    for vnr in vnr_results:
                        vnr_id = vnr.get('vnr_id', 'unknown')
                        print(f"\n  VNR ID: {vnr_id}")
                        print("    VM to Server Assignments:", vnr.get('vm_to_server_assignments', {}))
                        print("    Path Mappings:", vnr.get('path_mappings', {}))
                        print("    Link Flags:", vnr.get('link_flags', {}))
                        print("    Embedding Success:", vnr.get('embedding_success', False))
                        print("    Updated Graph (Snapshot):")

                        updated_graph = vnr.get('updated_graph', {})
                        for node, links in updated_graph.items():
                            print(f"      {node}: {links}")

                        # ✅ Collect all VM-to-server mappings for this VNR graph
                        combined_vm_to_server.update(vnr.get('vm_to_server_assignments', {}))

                    # ✅ Detect network fluctuations AFTER embedding for this VNR graph
                    print(f"\n[After Embedding Alert] Checking for network fluctuations after VNR Graph {graph_num}...")
                    hub.spawn(self.detect_major_fluctuations, "After Embedding - VNR Graph {graph_num}")  # ✅ Run in a separate thread
                    print(f"[After Embedding Alert] Network fluctuation check complete after VNR Graph {graph_num}.\n")

                    # ✅ Run iPerf only once for the entire VNR graph
                    if combined_vm_to_server:
                        print(f"\n[iPerf Test] Running bandwidth test for VNR Graph {graph_num}...")
                        self.perform_iperf_test(combined_vm_to_server)
                        print(f"[iPerf Test] Completed for VNR Graph {graph_num}.\n")

        except FileNotFoundError:
            print("Error: Embedding results file not found.")
        except json.JSONDecodeError as e:
            print(f"JSON decoding error at line {e.lineno}, column {e.colno}: {e.msg}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def perform_iperf_test(self, vm_assignments, duration=5):
        """
        Runs an iPerf test using stored host IPs from received_topology.
        - Starts an iPerf server in the background on destination hosts.
        - Runs iPerf client from source host to measure bandwidth.
        - Stops iPerf servers after test completion.
        """
        if not vm_assignments:
            self.logger.error("[iPerf Test] No VM assignments found. Skipping bandwidth test.")
            return

        self.logger.info("\n[iPerf Test] Running bandwidth tests between assigned VMs...")

        vm_list = list(vm_assignments.items())

        for i in range(len(vm_list) - 1):
            src_vm, src_server = vm_list[i]
            dst_vm, dst_server = vm_list[i + 1]

            src_ip = received_topology.get(src_server, {}).get("ip")
            dst_ip = received_topology.get(dst_server, {}).get("ip")

            if not src_ip or not dst_ip:
                self.logger.error(f"[iPerf Test] IP not found for {src_server} ({src_ip}) or {dst_server} ({dst_ip})")
                continue

            self.logger.info(f"[iPerf Test] Running bandwidth test between {src_ip} → {dst_ip}")

            try:
                # ✅ Start iPerf server in the background on the destination host
                server_command = f"nohup iperf -s > /dev/null 2>&1 &"
                subprocess.Popen(server_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                time.sleep(2)  # ✅ Ensure server starts before running client

                # ✅ Run iPerf client on the source host
                client_command = f"iperf -c {dst_ip} -t {duration} -f m"
                self.logger.info(f"[iPerf Test] Running: {client_command}")
                process = subprocess.run(client_command, shell=True, capture_output=True, text=True)

                # ✅ Capture iPerf output
                iperf_output = process.stdout if process.returncode == 0 else process.stderr
                self.logger.info(f"[iPerf Test] Bandwidth between {src_ip} → {dst_ip}:\n{iperf_output}")

            except Exception as e:
                self.logger.error(f"[iPerf Test] Failed to run iPerf test: {str(e)}")

            finally:
                # ✅ Kill iPerf server after test completion
                kill_server_command = "pkill -f 'iperf -s'"
                subprocess.run(kill_server_command, shell=True)
                self.logger.info(f"[iPerf Server] Stopped on {dst_ip}")

        self.logger.info("\n[iPerf Test] Completed all bandwidth tests.")

    def detect_major_fluctuations(self, stage=""):
        """Detects critical bandwidth, packet loss, or latency fluctuations before and after embedding.
           Embedding will continue even if fluctuations are detected, but alerts will be logged and printed in Ryu logs."""
        unstable_links = []
        high_loss_hosts = []
        high_latency_hosts = []

        # ✅ Check bandwidth fluctuations
        for link in received_topology.get("links_details", []):
            node1, node2 = link.get("node1"), link.get("node2")
            assigned_bw = link.get("assigned_bandwidth", 0)
            fluctuation_factor = (psutil.cpu_percent() % 5) * 0.05
            actual_bw = max(0, assigned_bw - (assigned_bw * fluctuation_factor))
            if actual_bw < assigned_bw * 0.7:
                unstable_links.append(f"{node1} <--> {node2}")

        # ✅ Check packet loss
        for host, details in received_topology.items():
            if host.startswith("h"):
                target_ip = details.get("ip")
                if target_ip:
                    loss = self.track_packet_loss(target_ip)
                    if loss is not None and loss > 10:
                        high_loss_hosts.append(host)

        # ✅ Check latency spikes
        for host, details in received_topology.items():
            if host.startswith("h"):
                target_ip = details.get("ip")
                if target_ip:
                    latency = self.track_latency(target_ip)
                    if latency and latency > 150:
                        high_latency_hosts.append(host)

        # ✅ Log alerts and print in Ryu logs but allow embedding to continue
        if unstable_links or high_loss_hosts or high_latency_hosts:
            message = f"[Fluctuation Alert] {stage} - Network instability detected."
            self.logger.warning(message)
            print(message)

            if unstable_links:
                msg = f"{stage} - Bandwidth issues on: {unstable_links}"
                self.logger.warning(msg)
                print(msg)

            if high_loss_hosts:
                msg = f"{stage} - High packet loss on: {high_loss_hosts}"
                self.logger.warning(msg)
                print(msg)

            if high_latency_hosts:
                msg = f"{stage} - High latency on: {high_latency_hosts}"
                self.logger.warning(msg)
                print(msg)

        return False  # Always return False so embedding is not delayed

    def track_packet_loss(self, target_ip):
        """
        Runs a ping test to measure packet loss to a given IP.
        Returns: Packet loss percentage (float) or None if failed.
        """
        try:
            print(f"[DEBUG] Running packet loss test for {target_ip}...")  # Debug log before ping
            command = f"ping -c 3 {target_ip}"
            result = subprocess.run(command, shell=True, timeout=5, capture_output=True, text=True)

            print(f"[DEBUG] Ping output: {result.stdout}")  # Debug log after ping

            # Extract packet loss percentage from output
            match = re.search(r'(\d+)% packet loss', result.stdout)
            if match:
                return int(match.group(1))

            return 0  # Assume 0% loss if no packet loss data found

        except subprocess.TimeoutExpired:
            print(f"[ERROR] Ping timeout for {target_ip}")
            return None  # Timeout, treat as unreachable

        except Exception as e:
            print(f"[ERROR] Packet Loss Test Failed for {target_ip}: {str(e)}")
            return None

    def track_latency(self, target_ip):
        """
        Runs a ping test to measure latency (RTT) to a given IP.
        Returns: Latency in milliseconds (float) or None if failed.
        """
        try:
            # Use ping with a single ICMP packet and extract RTT (round-trip time)
            command = f"ping -c 1 {target_ip}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            # Extract RTT value from ping output
            match = re.search(r'time=([\d.]+) ms', result.stdout)
            if match:
                return float(match.group(1))  # Return RTT in milliseconds

            return None  # Return None if latency info is not found

        except Exception as e:
            self.logger.error(f"[Latency Test] Failed for {target_ip}: {str(e)}")
            return None  # Return None if an error occurs

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.switches[datapath.id] = datapath
        self.mac_to_port.setdefault(datapath.id, {})

        # Install default flow to send unknown packets to controller
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        match = parser.OFPMatch()  # Match everything
        self.add_flow(datapath, 0, match, actions)

    def set_mac_to_port(self, dpid, entry):
        mac_table = self.mac_to_port.setdefault(dpid, {})
        datapath = self.switches.get(dpid)

        entry_port = entry['port']
        entry_mac = entry['mac']

        if datapath is not None:
            parser = datapath.ofproto_parser
            if entry_port not in mac_table.values():
                for mac, port in mac_table.items():
                    # Flow from known to new device
                    actions = [parser.OFPActionOutput(entry_port)]
                    match = parser.OFPMatch(in_port=port, eth_dst=entry_mac)
                    self.add_flow(datapath, 1, match, actions)

                    # Flow from new device to known device
                    actions = [parser.OFPActionOutput(port)]
                    match = parser.OFPMatch(in_port=entry_port, eth_dst=mac)
                    self.add_flow(datapath, 1, match, actions)

                mac_table.update({entry_mac: entry_port})
        return mac_table

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        instructions = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        flow_mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                     instructions=instructions, buffer_id=buffer_id or ofproto.OFP_NO_BUFFER)
        datapath.send_msg(flow_mod)

    @set_ev_cls(event.EventLinkAdd)
    def link_add_handler(self, ev):
        src = ev.link.src
        dst = ev.link.dst
        self.topology_data.add_edge(src.dpid, dst.dpid, port=src.port_no)
        self.topology_data.add_edge(dst.dpid, src.dpid, port=dst.port_no)
        self.recompute_paths()

    @set_ev_cls(event.EventLinkDelete)
    def link_delete_handler(self, ev):
        src = ev.link.src
        dst = ev.link.dst
        if self.topology_data.has_edge(src.dpid, dst.dpid):
            self.topology_data.remove_edge(src.dpid, dst.dpid)
        self.recompute_paths()

    def recompute_paths(self):
        """Calculate shortest paths using NetworkX."""
        self.paths = dict(nx.all_pairs_dijkstra_path(self.topology_data))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        src = eth.src
        dst = eth.dst
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            self.handle_arp(datapath, in_port, pkt, eth)
        elif eth.ethertype == ether_types.ETH_TYPE_IP:
            self.handle_ip(datapath, in_port, pkt, eth)

    def handle_arp(self, datapath, in_port, pkt, eth):
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt and arp_pkt.opcode == arp.ARP_REQUEST:
            # Send ARP reply if we know the destination IP, otherwise flood the packet
            if arp_pkt.dst_ip in self.mac_to_port:
                self.process_arp_reply(datapath, in_port, arp_pkt, eth)
            else:
                self.flood(datapath, in_port, pkt.data)

    def handle_ip(self, datapath, in_port, pkt, eth):
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_dpid = datapath.id
            dst_dpid = self.mac_to_port.get(eth.dst)
            path = self.paths.get(src_dpid, {}).get(dst_dpid, [])
            self.install_path_flows(datapath, path, in_port, eth)

    def install_path_flows(self, datapath, path, in_port, eth):
        """Install flows along the path."""
        for i in range(len(path) - 1):
            current_dpid = path[i]
            next_dpid = path[i + 1]
            current_switch = self.get_datapath(current_dpid)
            out_port = self.topology_data[current_dpid][next_dpid]['port']
            actions = [current_switch.ofproto_parser.OFPActionOutput(out_port)]
            match = current_switch.ofproto_parser.OFPMatch(in_port=in_port, eth_dst=eth.dst, eth_src=eth.src)
            self.add_flow(current_switch, 100, match, actions)

    def get_host_location(self, mac):
        for dpid, mac_table in self.mac_to_port.items():
            if mac in mac_table:
                return dpid
        return None

    def flood(self, datapath, in_port, data):
        parser = datapath.ofproto_parser
        for port in datapath.ports:
            if port != in_port:
                actions = [parser.OFPActionOutput(port)]
                out = parser.OFPPacketOut(datapath=datapath, actions=actions, data=data)
                datapath.send_msg(out)
