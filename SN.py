import sys
import pickle
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.node import RemoteController
import random
from randomPoissonDistribution import randomPoissonNumber_rand as randomPoissonNumber
import argparse
import numpy as np
import requests
import threading

class MyCustomTopo(Topo):
    "Custom topology example with spine-leaf architecture."

    def __init__(self, num_spine_switches, num_leaf_switches, num_hosts, min_cpu_range, max_cpu_range,
                 bandwidth_range_host_leaf, bandwidth_range_leaf_spine, ch):
        "Create custom topology."
        Topo.__init__(self)

        self.num_spine_switches = num_spine_switches
        self.num_leaf_switches = num_leaf_switches
        self.num_hosts = num_hosts

        # Number of spines, leafs, and servers per domain
        spines_per_domain = 1
        leafs_per_spine = 6
        servers_per_leaf = 3

        # Add spine and leaf switches
        spines = [self.addSwitch(f's{i}') for i in range(1, num_spine_switches + 1)]
        leaves = [self.addSwitch(f'l{i}') for i in range(1, num_leaf_switches + 1)]
        created_links = []

        # Choose distribution type based on `ch`
        if ch == 1:
            # Random CPU assignment
            hosts = [
                self.addHost(f'h{i}', ip=f'127.0.{i // 5 + 1}.{i % 5 + 2}', defaultRoute=f'via 127.0.{i // 5 + 1}.254',
                             cpu=random.randint(min_cpu_range, max_cpu_range)) for i in range(1, num_hosts + 1)]
        elif ch == 2:
            # Uniform CPU assignment
            hosts = [
                self.addHost(f'h{i}', ip=f'127.0.{i // 5 + 1}.{i % 5 + 2}', defaultRoute=f'via 127.0.{i // 5 + 1}.254',
                             cpu=random.uniform(min_cpu_range, max_cpu_range)) for i in range(1, num_hosts + 1)]
        elif ch == 3:
            # Normal distribution for CPU limits
            hosts = [
                self.addHost(f'h{i}', ip=f'127.0.{i // 5 + 1}.{i % 5 + 2}', defaultRoute=f'via 127.0.{i // 5 + 1}.254',
                             cpu=max(1, int(np.random.normal(min_cpu_range, max_cpu_range)))) for i in
                range(1, num_hosts + 1)]
        else:
            # Poisson distribution
            hosts = [
                self.addHost(f'h{i}', ip=f'127.0.{i // 5 + 1}.{i % 5 + 2}', defaultRoute=f'via 127.0.{i // 5 + 1}.254',
                             cpu=randomPoissonNumber(min_cpu_range, max_cpu_range, 0.4)) for i in
                range(1, num_hosts + 1)]

        # Linking spines, leaves, and hosts
        host_idx = 0
        for leaf in leaves:
            # Connect each leaf to all spine switches
            for spine in spines:
                bw = int(random.uniform(bandwidth_range_leaf_spine[0],
                                        bandwidth_range_leaf_spine[1])) if ch != 1 else random.randint(
                    bandwidth_range_leaf_spine[0], bandwidth_range_leaf_spine[1])
                self.addLink(leaf, spine, bw=bw)
                created_links.append((leaf, spine, bw))

            # Connect exactly 3 servers to each leaf
            domain_hosts = hosts[host_idx:host_idx + servers_per_leaf]
            host_idx += servers_per_leaf
            for host in domain_hosts:
                bw = int(random.uniform(bandwidth_range_host_leaf[0],
                                        bandwidth_range_host_leaf[1])) if ch != 1 else random.randint(
                    bandwidth_range_host_leaf[0], bandwidth_range_host_leaf[1])
                self.addLink(leaf, host, bw=bw)
                created_links.append((leaf, host, bw))

        self.created_links = created_links


    def print_link_details(self):
        info("\nSubstrate Network Physical Link details:\n")
        for link in self.created_links:
            node1, node2, assigned_bw = link
            info(
                f"Link {node1.name if hasattr(node1, 'name') else node1} - {node2.name if hasattr(node2, 'name') else node2} with bandwidth {int(assigned_bw)}\n")

def dumpNodeConnectionsToPickle(hosts, topo, pickle_file):
    "Dump connections to/from all nodes to a pickle file."
    data = {
        'num_spine_switches': topo.num_spine_switches,
        'num_leaf_switches': topo.num_leaf_switches,
        'num_hosts': topo.num_hosts,
        'links_details': []
    }

    for link in topo.created_links:
        node1, node2, assigned_bw = link
        link_details = {
            'node1': node1.name if hasattr(node1, 'name') else node1,
            'node2': node2.name if hasattr(node2, 'name') else node2,
            'assigned_bandwidth': int(assigned_bw)
        }
        data['links_details'].append(link_details)

    info("\nSubstrate Network Host's CPU and IP details:\n")
    for host in hosts:
        host_data = {
            'allocated_cores': round(host.params['cpu']),
            'connections': [],
            'ip': host.IP()  # Store the assigned IP
        }
        for conn in host.connectionsTo(hosts):
            if conn[0].isSwitch():
                assigned_bw = int(conn[0].connectionsTo(conn[1])[0].status.bw)
                link_data = {
                    'switch_name': conn[0].name,
                    'bandwidth': assigned_bw if assigned_bw is not None else 'Not Assigned'
                }
                host_data['connections'].append(link_data)

        data[host.name] = host_data  # Save host details including IP

    topo.print_link_details()

    # Dump the data to the pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Topology with Host IPs saved to {pickle_file}")

def send_sn_topology_to_ryu(data):
    url = 'http://127.0.0.1:8080/topology'
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            info("SN topology details successfully sent to Ryu controller.\n")
        else:
            info(f"Failed to send SN topology. Status Code: {response.status_code}\n")
    except Exception as e:
        info(f"Error sending SN topology to Ryu: {e}\n")


def runExperimentToPickle(num_spine_switches, num_leaf_switches, num_hosts, min_cpu_range, max_cpu_range,
                          bandwidth_range_host_leaf, bandwidth_range_leaf_spine, ch, pickle_file):
    info("Starting network setup...\n")
    topo = MyCustomTopo(num_spine_switches, num_leaf_switches, num_hosts, min_cpu_range, max_cpu_range,
                        bandwidth_range_host_leaf, bandwidth_range_leaf_spine, ch)
    topo.print_link_details()
    net = Mininet(topo=topo, controller=None)
    ryu_controller = RemoteController('c0', ip='127.0.0.1', port=6653)
    net.addController(ryu_controller)

    try:
        net.start()
        info("Network started, running pingAll to verify connectivity...\n")
        net.pingAll()
        info("pingAll completed. Network is operational and managed by Ryu in real-time.\n")

        # Save topology details to pickle file
        dumpNodeConnectionsToPickle(net.hosts, topo, pickle_file)

        # Send SN topology to Ryu controller
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        send_sn_topology_to_ryu(data)  # Send data to Ryu after saving it to the pickle file

    except Exception as e:
        info(f"Error during Mininet setup or connectivity test: {e}")
    finally:
        net.stop()
        info("Network stopped, experiment completed.\n")

if __name__ == '__main__':
    if len(sys.argv) != 12:
        print(
            "Usage: python3 Mininet.py <num_spine_switches> <num_leaf_switches> <num_hosts> <min_cpu_range> <max_cpu_range> <bw_host_leaf_min> <bw_host_leaf_max> <bw_leaf_spine_min> <bw_leaf_spine_max> <ch> <pickle_file>")
        sys.exit(1)

    num_spine_switches = int(sys.argv[1])
    num_leaf_switches = int(sys.argv[2])
    num_hosts = int(sys.argv[3])
    min_cpu_range = int(sys.argv[4])
    max_cpu_range = int(sys.argv[5])
    bw_host_leaf_min = int(sys.argv[6])
    bw_host_leaf_max = int(sys.argv[7])
    bw_leaf_spine_min = int(sys.argv[8])
    bw_leaf_spine_max = int(sys.argv[9])
    ch = int(sys.argv[10])
    pickle_file = sys.argv[11]
    bandwidth_range_host_leaf = (bw_host_leaf_min, bw_host_leaf_max)
    bandwidth_range_leaf_spine = (bw_leaf_spine_min, bw_leaf_spine_max)

    setLogLevel('info')
    runExperimentToPickle(num_spine_switches, num_leaf_switches, num_hosts, min_cpu_range, max_cpu_range,
                          bandwidth_range_host_leaf, bandwidth_range_leaf_spine, ch, pickle_file)

