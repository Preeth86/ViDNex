# ViDNex: Ranking-Based Virtual Data Center Embedding for Joint Load Balancing and Energy Efficiency in Next Generation Infrastructure

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
PyCharm Community Edition 2022. <br />
Mininet for PN & VDCR topology generation.<br />
Introduction about VNE problem can be found in below link:<br />
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


