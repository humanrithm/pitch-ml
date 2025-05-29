`pitch-ml`$\textbf{: Optimizing Pitcher Health and Performance}$

Ball tracking, biomechanics, and machine learning algorithms for optimizing pitcher health and performance. 

$\textbf{1. Repository Overview}$

- `.bin`: AWS connection configuration.
- `dev`: Any development tasks (e.g., biomechanics, data science), sorted by category and project.
- `prod`: Research- or production-ready code, usually adapted from `dev`. 
- `packages`: Modules & functions that can be accessed within the repo. Note that these currently require running `pip install -e .` to access. 
- `qa`: Debugging or revision-specific tasks (e.g., biomechanics, data science), sorted by categoory and project. 

$\textbf{A.1 Environment Setup}$

All code is configured to be run in a conda virtual environment (details in `pitch_ml.yml`) from Python `3.11.10`, which has all OpenSim API dependencies installed. To activate the environment, call `conda activate pitch_ml`. 

$\textbf{A.2 AWS Connection}$

For AWS connections, make sure to run: 

- (1) Ensure executable permissions:
    - `chmod +x .bin/update_ip.sh`
    - `chmod +x .bin/tunnel.sh`
- (2) Run scripts in terminal (from root directory):
    - `./.bin/update_ip.sh`: Updates the EC2 IP address (if in a new connection)
    - `./.bin/tunnel.sh`: Creates a secure shell (SSH) tunnel to the EC2 instance; this enables local connection to the RDS

More details can be found in the `.bin` folder.

$\textbf{A.3 Publications}$

- __Moore, R.C., Gurchiek, R.D. & Avedesian, J.M.__ `A context-enhanced deep learning approach to predict baseball pitch location from ball tracking release metrics.` _Sports Engineering_ 28, 16 (2025). https://doi.org/10.1007/s12283-025-00497-5
    - Relevant repository sections: __Coming Soon__.
