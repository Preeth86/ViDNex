# ViDNex: Ranking-Based Virtual Data Center Embedding for Joint Load Balancing and Energy Efficiency in Next Generation Infrastructure

## Overview

**ViDNex** is an advanced **Virtual Data Center Embedding (VDCE)** framework that leverages **ranking-based** methods to optimize **load balancing** and **energy minimization** in next-generation infrastructures. It incorporates **AHP** (Analytic Hierarchy Process) and **VIKOR** (VlseKriterijumska Optimizacija I Kompromisno Resenje) ranking techniques for **VM-to-host** assignment, alongside system-level parameters such as **CRB utilization**, **energy consumption**, and **load balance**.

## Features

- **Ranking-based Embedding**: Utilizes **AHP** and **VIKOR** for intelligent VM-to-host assignment, considering system-level parameters.
- **Multiple Embedding Strategies**: Includes multiple baseline embedding strategies such as **CEVNE**, **DROI**, **First Fit**, and others, with **ViDNex** serving as the optimized strategy.
- **Load Balancing**: Ensures efficient load distribution across hosts while minimizing energy consumption.
- **Energy Minimization**: Focuses on reducing the overall energy consumption of the data center.
- **Scalability**: Efficiently handles large-scale infrastructures with multiple Virtual Machines (VMs) and Hosts (physical machines).
- **Rollback Mechanisms**: Implements rollback support for both node and link embedding for error recovery.

## Installation

To use **ViDNex**, simply clone the repository:

```bash
git clone https://github.com/Preeth86/ViDNex.git
cd ViDNex
