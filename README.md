# ViDNex: Ranking-Based Virtual Data Center Embedding for Load Balancing and Energy Efficiency in Next-Generation Infrastructure

## Overview

**ViDNex** is an advanced **Virtual Data Center Embedding (VDCE)** framework that leverages **ranking-based** methods to optimize **load balancing** and **energy minimization** in next-generation infrastructures. It incorporates **AHP** (Analytic Hierarchy Process) and **VIKOR** (VlseKriterijumska Optimizacija I Kompromisno Resenje) ranking techniques for **VM-to-host** assignment, alongside system-level parameters such as **CRB utilization**, **energy consumption**, and **load balance**.

## Execution Environment:

**Operating System**: 64-bit Ubuntu 22.04.5 LTS.<br />
**Host**: Dell PowerEdge R750 server.<br />
**Physical Memory (RAM)**: 128 GB.<br />
**CPU**: Intel Xeon Gold 5317 @ 3.00 GHz.<br />
**Tools and Libraries**: Mininet, SDN-Ryu, OpenFlow protocol.<br />

### Prerequisites

Python 3.7 & above.<br />
PyCharm Community Edition 2025.<br />
Mininet for PN & VDCR topology generation.<br />
Ryu an SDN controller for network management.<br />
Introduction about the VNE problem can be found in the link below:<br />
https://www.youtube.com/watch?v=JKB3aVyCMuo&t=506s<br />

## Features

- **Ranking-based Embedding**: Utilizes **AHP** and **VIKOR** for intelligent VM-to-host assignment, considering system-level parameters.
- **Multiple Embedding Strategies**: Includes multiple baseline embedding strategies such as **CEVNE**, **DROI**, **First Fit**, and others, with **ViDNex** serving as the optimized strategy.
- **Load Balancing**: Ensures efficient load distribution across hosts while minimizing energy consumption.
- **Energy Minimization**: Focuses on reducing the overall energy consumption of the data center.
- **Scalability**: Efficiently handles large-scale infrastructures with multiple Virtual Machines (VMs) and Hosts (physical machines).
- **Rollback Mechanisms**: Implements rollback support for both node and link embedding for error recovery.

## Installation

###   Download  ViDNex and keep it in the drive. The ViDNex file contains all executable files related to the proposed and baseline approaches. <br />

- manager.py -> The Main file related to the ViDNex broker.<br />
- Ryu.py -> The Main file related to the SDN-Ryu controller.<br />
- SN.py -> The Main file related to the Physical Network topology generation. <br /> 
- VNR.py -> The Main file related to the VDCR topology generation. <br />
- ViDNex.py -> The Main file related to the proposed ViDNex approach. <br />
- CEVNE.py -> The Main file related to the CEVNE baseline approach. <br />
- DROI.py -> The Main file related to the DROI baseline approach. <br />
- First_Fit.py -> The Main file related to the First Fit baseline approach. <br />
- LitE.py -> The Main file related to the LitE baseline approach. <br />
- SCA-R.py -> The Main file related to the SCA-R baseline approach. <br />

## How to Execute the Framework

Follow the step-by-step instructions below to run the **ViDNex** framework:

### 1. **Setup Mininet and Ryu**

**Step 1.1**: Install **Mininet** (network emulator) and **Ryu** (SDN controller).<br />

sudo apt-get update <br />
sudo apt-get install mininet <br />
pip install ryu <br />

### 2. **Prepare the Virtual Environment**
**Step 2.1**: Create a virtual environment for Python dependencies.<br />
python3 -m venv .venv
source .venv/bin/activate

**Step 2.2**: Install the required Python libraries.<br />
pip install pulp <br />
pip install scikit-learn <br />
pip install networkx <br />
pip install matplotlib <br />
pip install numpy <br />
pip install psutil <br />
pip install openpyxl <br />

**Step 2.3**: Start the Ryu controller.<br />
ryu-manager ryu.app.simple_switch_13

###  In VNR.py, we can set the various parameters related to Virtual data center requests(VDCRs).<br />

- We can set the minimum and maximum number of VMs of VDCRs in the execute_vnr function. <br />
- We can set the VDCR demands like CRB(min, max), BandWidth(min, max). <br />
- Example: (2, 10, 1, 10, 1, 5) <br />

- Run VNR.py after making any modifications. <br />

###  In SN.py, we can set the various parameters related to Physical Network (PN).<br />

- We can set the number of spine switches, per spine switch 6 leaf switches, and per leaf switch 3 hosts in PN in the execute_substrate function.<br />
- We can set the PN available resources like CRB(min, max), BandWidth(min, max) host to leaf, BandWidth(min, max) leaf to spine. <br />
- Example: (3, 18, 54, 1000, 2000, 5500, 6500, 5500, 6500) <br />

- Run SN.py after making any modifications. <br />

###  In manager.py:<br />

- In the manager.py, select the PN and VM distribution, then select the VDCR distribution and then the number of VDCRs and finally the embedding algorithm ViDNex to start embedding <br />

- This file generates the pickle and JSON files, which contain all the information about physical network topologies, such as the number of hosts, links, and connectivity. It also includes values for each physical network resource.

### In the manager.py file, set the VDCR size such as [200, 400, 600, 800, 1000], and also by default, 10 iterations will be executed for each VDCR size in the iteration variable.<br />

- Finally, once the manager.py runs. After successfully running, the Final embedding results are captured in Results.xlsx, which includes values for various performance metrics for all test scenarios and for every iteration.

## Contributors
- Mr. N Preetham <br />
https://scholar.google.com/citations?user=z_TrEuIAAAAJ&hl=en&oi=ao <br />
- Dr. Sourav Kanti Addya <br />
https://souravkaddya.in/ <br />
- Dr. Keerthan Kumar T G<br />
https://scholar.google.com/citations?user=fW7bzK8AAAAJ&hl=en <br />
- Dr Saumya Hegde <br />
https://scholar.google.co.in/citations?user=WAyKHHwAAAAJ&hl=en <br />


## Contact
If you have any questions, simply write a mail to  preetham(DOT)nagaraju(AT)gmail(DOT)com.
