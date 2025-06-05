import subprocess

# Run the Ryu controller with the path to your Ryu.py file in the project directory
subprocess.run([
    "/home/vnesdn/PycharmProjects/EFraS++/.venv/bin/ryu-manager",  # Full path to ryu-manager
    "/home/vnesdn/PycharmProjects/EFraS++/Ryu.py",  # Now the path is inside the project directory
    "ryu.app.ofctl_rest",  # REST API for OpenFlow control
    "ryu.app.rest_topology"  # REST API for topology data
])
