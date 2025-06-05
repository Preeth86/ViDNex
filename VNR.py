import pickle
import argparse
import random
import numpy as np
from randomPoissonDistribution import randomPoissonNumber_rand as randomPoissonNumber

# def generate_vne_requests(num_requests, vm_range, cpu_range, bw_range, ch, vl_prob):
#     vne_requests = []
#
#     for p in range(num_requests):
#         if ch == 1:
#             num_vms = random.randint(*vm_range)
#             vm_cpu_cores = [random.randint(*cpu_range) for _ in range(num_vms)]
#         elif ch == 2:
#             num_vms = int(random.uniform(*vm_range))
#             vm_cpu_cores = [int(random.uniform(*cpu_range)) for _ in range(num_vms)]
#         elif ch == 3:
#             num_vms = max(1, int(np.random.normal(np.mean(vm_range), np.std(vm_range))))
#             vm_cpu_cores = [max(1, int(np.random.normal(np.mean(cpu_range), np.std(cpu_range)))) for _ in range(num_vms)]
#         elif ch == 4:
#             num_vms = int(randomPoissonNumber(vm_range[0], vm_range[1], 0.4))
#             vm_cpu_cores = [int(randomPoissonNumber(cpu_range[0], cpu_range[1], 0.4)) for _ in range(num_vms)]
#         else:
#             raise ValueError("Invalid choice for generating VNR")
#
#         lst = [(i + 1, j + 1) for i in range(num_vms) for j in range(i + 1, num_vms)]
#         y = (num_vms * (num_vms - 1)) / 2
#         x = max(1, int(y * vl_prob))
#         x = int(y * vl_prob)  # Set virtual links based on selected probability
#         vm_links = random.sample(lst, min(x, len(lst)))
#         bandwidth_values = [random.randint(*bw_range) for _ in range(len(vm_links))]
#
#         vne_request = {
#             'num_vms': num_vms,
#             'vm_cpu_cores': vm_cpu_cores,
#             'vm_links': vm_links,
#             'bandwidth_values': bandwidth_values,
#             'vnr_id': p + 1
#         }
#
#         vne_requests.append(vne_request)
#         # Print the output for the current VNE request
#         print(f"\nRequest {len(vne_requests)}:")
#         print(f"  Number of VMs: {num_vms}")
#         print(f"  VM CPU Cores: {vm_cpu_cores}")
#         print("  Virtual Links established:")
#         for i, link in enumerate(vm_links, 1):
#             print(f"    Link {i}: VM{link[0]} - VM{link[1]}")
#         print("  Virtual Links Bandwidth Demand:")
#         for i, bandwidth in enumerate(bandwidth_values, 1):
#             print(f"    Link {i}: {bandwidth}")
#
#     return vne_requests

def generate_vne_requests(num_requests, vm_range, cpu_range, bw_range, ch, vl_prob):
    vne_requests = []

    for p in range(num_requests):
        if ch == 1:
            num_vms = random.randint(*vm_range)
            vm_cpu_cores = [random.randint(*cpu_range) for _ in range(num_vms)]
        elif ch == 2:
            num_vms = int(random.uniform(*vm_range))
            vm_cpu_cores = [int(random.uniform(*cpu_range)) for _ in range(num_vms)]
        elif ch == 3:
            num_vms = max(1, int(np.random.normal(np.mean(vm_range), np.std(vm_range))))
            vm_cpu_cores = [max(1, int(np.random.normal(np.mean(cpu_range), np.std(cpu_range)))) for _ in range(num_vms)]
        elif ch == 4:
            num_vms = int(randomPoissonNumber(vm_range[0], vm_range[1], 0.4))
            vm_cpu_cores = [int(randomPoissonNumber(cpu_range[0], cpu_range[1], 0.4)) for _ in range(num_vms)]
        else:
            raise ValueError("Invalid choice for generating VNR")

        # Calculate total number of possible virtual links (edges)
        lst = [(i + 1, j + 1) for i in range(num_vms) for j in range(i + 1, num_vms)]
        y = (num_vms * (num_vms - 1)) // 2  # Total number of possible links (fully connected graph)

        # Calculate the number of links based on vl_prob
        x = max(1, int(y * vl_prob))

        # Ensure the graph is connected by using a spanning tree if necessary
        if x < num_vms - 1:  # Less than the minimum number of links needed for a connected graph
            x = num_vms - 1  # Set to the minimum number of links (spanning tree)

        # Select the required number of links randomly
        vm_links = random.sample(lst, min(x, len(lst)))

        # Generate random bandwidth values for the selected links
        bandwidth_values = [random.randint(*bw_range) for _ in range(len(vm_links))]

        vne_request = {
            'num_vms': num_vms,
            'vm_cpu_cores': vm_cpu_cores,
            'vm_links': vm_links,
            'bandwidth_values': bandwidth_values,
            'vnr_id': p + 1
        }

        vne_requests.append(vne_request)

        # Print the output for the current VNE request
        print(f"\nRequest {len(vne_requests)}:")
        print(f"  Number of VMs: {num_vms}")
        print(f"  VM CPU Cores: {vm_cpu_cores}")
        print("  Virtual Links established:")
        for i, link in enumerate(vm_links, 1):
            print(f"    Link {i}: VM{link[0]} - VM{link[1]}")
        print("  Virtual Links Bandwidth Demand:")
        for i, bandwidth in enumerate(bandwidth_values, 1):
            print(f"    Link {i}: {bandwidth}")

    return vne_requests


def save_vne_requests_to_pickle(vne_requests, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(vne_requests, f)

def main():
    parser = argparse.ArgumentParser(description='Virtual Network Embedding Generator')
    parser.add_argument('mininet_pickle_file', help='Path to the Mininet pickle file')
    parser.add_argument('output_file', help='Output path for the VNE requests pickle file')
    parser.add_argument('num_requests', type=int, help='Number of VNE requests')
    parser.add_argument('vm_range', nargs=2, type=int, help='Number of VMs range')
    parser.add_argument('cpu_range', nargs=2, type=int, help='CPU core range for each VM')
    parser.add_argument('bw_range', nargs=2, type=int, help='Bandwidth range between VMs')
    parser.add_argument('ch', type=str, help='choice for the generating vnr')
    parser.add_argument('vl_prob', type=float, help='Virtual Link Connectivity Probability')
    args = parser.parse_args()

    ch = int(args.ch)
    vne_requests = generate_vne_requests(args.num_requests, args.vm_range, args.cpu_range, args.bw_range, ch, args.vl_prob)

    save_vne_requests_to_pickle(vne_requests, args.output_file)

if __name__ == '__main__':
    main()
