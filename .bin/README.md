$\textbf{User Bin Setup}$

__To Run__: 

- (1) Ensure executable permissions:
    - `chmod +x .bin/update_ip.sh`
    - `chmod +x .bin/tunnel.sh`
- (2) Run scripts in terminal (from root directory):
    - `./.bin/update_ip.sh`: Updates the EC2 IP address (if in a new connection)
    - `./.bin/tunnel.sh`: Creates a secure shell (SSH) tunnel to the EC2 instance; this enables local connection to the RDS