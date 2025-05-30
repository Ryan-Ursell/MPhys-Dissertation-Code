# MPhys-Dissertation-Code
This repository contains the code used in the analysis for the undergraduate MPhys project:

**_The Missing Multipole Problem: Biases in Gravitational Wave Analysis._**

## Project Overview
This project investigates how neglecting higher-order multipoles in waveform models can introduce biases in parameter estimation for binary black hole mergers, using Bayesian inference tools such as `bilby`.

The analyses were performed on the SCIAMA High Performance Computing (HPC) cluster using BILBY, with results stored as pickle files. This repository contains the Python code used to load these data files and to generate plots and perform bias quantification analyses.

## Setup & Requirements

### Required Packages
This project uses Python 3.10 ![Python Version](https://img.shields.io/badge/python-3.10-blue) and the following [dependencies](dependencies.txt)
```
Dependencies:
- `numpy - v1.24.3`
- `pandas - v2.0.2`
- `seaborn - v0.12.2`
- `matplotlib - v3.5.3`
- `scipy - v1.8.1`
- `bilby - v2.1.1`
- `pesummary - v0.13.10`
- Plus standard libraries `re`, `os`, `collections`
```

## Licence
This project is under the MIT ![License](https://img.shields.io/badge/license-MIT-green), see [LICENSE](LICENSE) for more infomation.

## Acknowledgements
Supervisors
- My supervisors, Charlie Hoy and Ian Harry, provided support and feedback throughout and were invaluable to the completion of this project

ChatGPT:
- GPT 4o was used to assist with debugging code used throughout the project

## Personal Information
Ryan Ursell  
ryanursell@outlook.com  
MPhys Physics, Astrophysics, and Cosmology  
University of Portsmouth  