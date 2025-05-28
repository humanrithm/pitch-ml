$\textbf{AWS Connection Steps}$

- __Step 1__: Run bash scripts (`.bin` folder in root directory)
    - Primarily serves to update IP address
- __Step 2__: Connect to EC2 instance in terminal
    - Can use this to troubleshoot EC2 functionality from terminal
- __Step 3__: Connect to AWS RDS in Python on port 5433
- __Step 4__: Connect to DB in VS Code using SQLTools
    - This also uses port 5433, hence the ordering