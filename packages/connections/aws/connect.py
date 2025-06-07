import os
import io
import boto3
import psutil
import psycopg2
import numpy as np
import pandas as pd
import psycopg2.extras
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from biomech.processing.trc import *                    # import TRC processing functions
from sshtunnel import SSHTunnelForwarder
from psycopg2.errors import UniqueViolation

class AWS():

    """
    **AWS**

    **Latest Version: `0.2.7`**\n
    **Author: `Connor Moore`**

    `v0.2.7` includes the ability to load XML files.

    AWS connection class to connect to an RDS database and S3 bucket via SSH tunnel.
    This class provides methods to connect to the database, run queries, upload data,
    and close the connection. It also handles SSH tunneling to securely connect to the
    RDS instance.

    Once connected, it is possible to query from the RDS using the `run_query` method,
    and upload data to a table using the `upload_data` method. The connection can be
    closed using the `close` method.

    S3 access is handled with an IAM role within the elastic EC2 instance.
    """

    __version__ = '0.2.7'

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
    # create folder within the S3 bucket
    def create_s3_folder(
            self, 
            folder_prefix: str
    ) -> None:
        """ Create folder (`folder_prefix`) in S3 if it doesn't already exist. """
        
        if not folder_prefix.endswith('/'):
            folder_prefix += '/'

        # check if any object exists with the given prefix
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=folder_prefix,
            MaxKeys=1
        )
        if 'Contents' in response:
            print(f"[AWS]: Folder s3://{self.bucket_name}/{folder_prefix} already exists.")
        else:
            # upload an empty object (simulates folder creation)
            self.s3.put_object(Bucket=self.bucket_name, Key=folder_prefix)
            print(f"[AWS]: Created folder s3://{self.bucket_name}/{folder_prefix}")

    def list_s3_objects(
            self, 
            prefix: str = '',
            file_type: str = None,
            paginate: bool = True
    ) -> list:
        """List files in the S3 bucket. Have the option to add a prefix and file type (e.g., .csv).

        **Args:**
            **prefix** (str): Prefix to filter the objects in the S3 bucket. Default is empty string, which lists all objects.
            **file_type** (str): File type to filter the objects (e.g., '.csv'). Default is None, which lists all objects.
            **paginate** (bool, default `True`): If True, will paginate through the results. Default is False, which returns up to 1,000 objects.
        """
        if paginate:
            paginator = self.s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            all_keys = []
            for page in page_iterator:
                contents = page.get('Contents', [])
                all_keys.extend(obj['Key'] for obj in contents)

            return all_keys

        else:
            # load full bucket
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            
            if file_type is not None:
                return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith(file_type)]
            else:
                return [obj['Key'] for obj in response.get('Contents', [])]
        
    def load_s3_object(
            self, 
            key: str,
            return_info: bool = True
    ) -> Union[tuple[bytes, dict], bytes]:
        """Load a specific object from the S3 bucket. Returns a Bytes object, which can be handled in a separate function, and relevant info extracted from file name."""
        
        # load object from S3 (applies to all types)
        response = self.s3.get_object(Bucket=self.bucket_name, Key=key)

        if return_info:
            # get info from key (+ check for static trial)
            if '@static' in key:
                key_info = {
                    'subject_id': key.split('/')[1],
                    'study_id': key.split('/')[1] + '_static'                                       # static trial
                }
            else:
                key_info = {
                    'subject_id': key.split('/')[1],
                    'study_id': key.split('/')[1] + '_' + key.split('_')[-1].split('.')[0]          # this gets the pitch number (e.g., _01)
                }

            return response['Body'].read(), key_info
        else:
            return response['Body'].read()    

    def load_xml_from_s3(
            self, 
            s3_key: str
    ) -> ET.Element:
        """Load an XML file from S3 and return its root element."""
        obj = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
        xml_bytes = obj['Body'].read()
        
        # decode and parse
        root = ET.fromstring(xml_bytes.decode('utf-8'))
        
        return root
 
    def upload_to_s3(
            self, 
            obj: Union[str, bytes, pd.DataFrame], 
            s3_key: str
    ) -> None:
        """Upload a Python object (string, bytes, or DataFrame) to S3. Uses `put_object` from S3 client to handle different data types.
        
        **Args:**
            **obj** (Union[str, bytes, pd.DataFrame]): The object to upload. Can be a string, bytes, or a pandas DataFrame.
            **s3_key** (str): The S3 key (path) where the object will be uploaded.
        **Raises:**
            **TypeError**: If the object type is not supported (not str, bytes, or pd.DataFrame).
        """
        
        # check data type
        if isinstance(obj, pd.DataFrame):
            buffer = io.StringIO()
            obj.to_csv(buffer, index=False)
            body = buffer.getvalue()
        elif isinstance(obj, str):
            body = obj
        elif isinstance(obj, bytes):
            body = obj
        else:
            raise TypeError("Unsupported type for upload. Supported types: str, bytes, pd.DataFrame.")
        
        # put object to S3
        self.s3.put_object(Bucket=self.bucket_name, Key=s3_key, Body=body)
        
        print(f"[AWS]: Uploaded object to s3://{self.bucket_name}/{s3_key}")
    
    def upload_trc_to_s3(self, header: list[str], body: pd.DataFrame, s3_key: str):
        """ Write TRC file (header + `DataFrame` body) directly to S3. """
        
        buffer = io.StringIO()

        # write header 7 body lines
        buffer.write("\n".join(header) + "\n")
        body.to_csv(buffer, sep="\t", index=False, header=False)

        # upload object to S3
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=buffer.getvalue()
        )
    
        print(f"[AWS]: TRC file written to s3://{self.bucket_name}/{s3_key}")
    
    """ RDS FUNCTIONS """
    # load all subject info
    def load_subject_info(self) -> pd.DataFrame:
        """ Load subject info from S3 bucket. Returns a CSV file with subject IDs and other relevant information."""
        subject_info_bytes = self.load_s3_object('subjects/summary/subject_info.csv', return_info=False)
        subject_info = pd.read_csv(io.BytesIO(subject_info_bytes))

        return subject_info
    
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