import os
import boto3
import psutil
import psycopg2
import numpy as np
import pandas as pd
import psycopg2.extras
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
from psycopg2.errors import UniqueViolation

class AWS():

    """
    **AWS**

    AWS connection class to connect to an RDS database and S3 bucket via SSH tunnel.
    This class provides methods to connect to the database, run queries, upload data,
    and close the connection. It also handles SSH tunneling to securely connect to the
    RDS instance.

    Once connected, it is possible to query from the RDS using the `run_query` method,
    and upload data to a table using the `upload_data` method. The connection can be
    closed using the `close` method.

    S3 access is handled with an IAM role within the elastic EC2 instance. Full access
    and retrieval is not implemented yet, but is pending in `v0.2.5`.
    """

    __version__ = '0.2.5'

    def __init__(self):
        self.connected = 0

        # get __path__ to this file and use it to load .env
        secrets_path = os.path.join(os.path.dirname(__file__), 'secrets_aws.env')
        self.creds = load_dotenv(secrets_path)

    # create connection do database
        # port: local port to connect to RDS (defaults to 5433)
    def connect(
            self,
            port: int = 5433
    ) -> None:
        self.tunnel = self.__setup_ssh(port)        # set up ssh tunnel

        """ RDS Connection"""
        # create connection to RDS
        self.connection = psycopg2.connect(
            host='localhost',
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PW'),
            port=self.tunnel.local_bind_port 
        )  
        self.connected = 1

        print('[AWS]: Connected to RDS endpoint.')

        """ S3 Bucket """
        self.s3 = boto3.client("s3")
        self.bucket_name = 'pitch-ml'

    """ S3 CONNECTIONS """
    def list_s3_objects(
            self, 
            prefix: str = ''
    ) -> list:
        """List files in the S3 bucket under an optional prefix."""
        
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        
        return [obj['Key'] for obj in response.get('Contents', [])]

    """ RDS FUNCTIONS """
    # run queries in database
    def run_query(
            self,
            query: str
    ) -> pd.DataFrame:
        if self.connected == 0:
            self.connect()
        
        return pd.read_sql_query(query, self.connection)
    
    # upload data to a table
    def upload_data(
            self, 
            data: pd.DataFrame, 
            table_name: str
    ) -> None:
        if self.connected == 0:
            self.connect()
        
        # create a cursor
        cursor = self.connection.cursor()
        
        # convert any numpy types to native python types
        data = data.applymap(lambda x: x.item() if isinstance(x, np.generic) else x)
        
        # create a list of tuples from the dataframe values & get columns
        tuples = [tuple(x) for x in data.to_numpy()]
        cols = ','.join([f'"{col}"' for col in data.columns])       # v0.2.3
        
        # create the SQL query to insert data
        query = f"""
            INSERT INTO {table_name} ({cols}) 
            VALUES %s;
        """
        
        try:
            psycopg2.extras.execute_values(cursor, query, tuples)
            self.connection.commit()
            print(f"[AWS]: Data uploaded to {table_name} successfully.")
        except UniqueViolation as e:  # primary key violation rollback (v0.2.4)
            print(f"[AWS]: Duplicate entry found. Skipping this row.")
            self.connection.rollback()
        except Exception as e:
            print(f"[AWS]: Error uploading data to {table_name}: {e}")
            self.connection.rollback()
        finally:
            cursor.close()
    
    """ CONNECTION MANAGEMENT """
    # close connection & tunnel
    def close(self):
        # database connection
        if self.connected:
            self.connection.close()
            print("[AWS]: Database connection closed.")
            
            # SSH tunnel
            if self.tunnel:
                self.tunnel.stop()
                print("[AWS]: SSH tunnel stopped.")
            
            self.connected = 0
        
        else:
            print("[AWS]: No active connection to close.")

    # set up ssh tunnel
    def __setup_ssh(self,
                    port: int = 5433):
        self.__kill_process_on_port(port)                           # check availability of port
        
        tunnel =  SSHTunnelForwarder(
            (os.getenv('EC2_IP'), 22),                               # EC2 public IP and port
            ssh_username=f'{os.getenv("EC2_USER")}',                 # EC2 username
            ssh_pkey=os.getenv('PRIVATE_KEY_PATH'),                  # path to SSH private key
            remote_bind_address=(os.getenv('AWS_ENDPOINT'), 5432),   # RDS endpoint and port
            local_bind_address=('localhost', 5433)
        )

        tunnel.start()      # start the tunnel

        return tunnel
    
    # check if port (5433) is in use and kill the process
        # debug: if you re-run .connect() in the same notebook, should kill the kernel
    def __kill_process_on_port(self, 
                               port: int):
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            connections = proc.info.get('connections')
            
            # check if process has connections
            if connections:
                for conn in connections:
                    if conn.laddr.port == port:
                        print(f"[AWS]: Port {port} is in use by process {proc.info['name']} (PID {proc.info['pid']}). Killing it.")
                        proc.kill()
                        
                        return
        
        print(f"[AWS]: Port {port} is free.")