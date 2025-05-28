`pitch-ml`$\textbf{: Optimizing Pitcher Health and Performance}$

Ball tracking, biomechanics, and machine learning algorithms for optimizing pitcher health and performance. 

$\textbf{1. Repository Overview}$

Coming soon.

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

Any publications associated with this repository will be linked here.
