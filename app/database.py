import mysql.connector
import os
from minio import Minio
from urllib.parse import urlparse

def create_requests_table():
    """
    Connects to MariaDB and creates the 'requests' table if it doesn't exist.
    """
    try:
        # Connect to MariaDB
        conn = mariadb_connect()
        cur = conn.cursor()

        # Create table query
        table_creation_query = """
            CREATE TABLE IF NOT EXISTS requests (
                uuid VARCHAR(36) PRIMARY KEY,
                status ENUM('WAITING', 'RUNNING', 'DONE', 'FAILED') NOT NULL,
                path_to_model VARCHAR(255) NOT NULL
            );
        """

        # Execute the query
        cur.execute(table_creation_query)

        # Commit the changes
        conn.commit()

        print("Table 'requests' created successfully or already exists.")

    except mysql.connector.Error as e:
        print(f"Error creating table: {e}")

    finally:
        if conn:
            cur.close()
            conn.close()


def mariadb_connect() -> mysql.connector.MySQLConnection:
    """
    Establishes a connection to MariaDB using Kubernetes service discovery.
    """
    try:
        # Use Kubernetes service environment variables
        mydb = mysql.connector.connect(
            user=os.environ.get("MARIADB_USER"),
            password=os.environ.get("MARIADB_PASSWORD"),
            host=os.environ.get("MARIADB_SERVICE_HOST"),  # Resolved by Kubernetes DNS
            port=int(os.environ.get("MARIADB_SERVICE_PORT")),  # Resolved by Kubernetes DNS
            database=os.environ.get("MARIADB_DATABASE")
        )
        print(f"Successfully connected to MariaDB at: {os.environ.get('MARIADB_SERVICE_HOST')}:{os.environ.get('MARIADB_SERVICE_PORT')}")
        return mydb
    except mysql.connector.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        raise

def minio_connect() -> Minio:
    """
    Establishes a connection to MinIO using Kubernetes service discovery.
    """
    try:
        # Use Kubernetes service environment variables
        endpoint = f"{os.environ.get('MINIO_SERVICE_HOST')}:{os.environ.get('MINIO_SERVICE_PORT')}"
        print(f"MinIO Endpoint: {endpoint}")

        minio_client = Minio(
            endpoint,
            access_key=os.environ.get("MINIO_ACCESS_KEY"),
            secret_key=os.environ.get("MINIO_SECRET_KEY"),
            secure=False  # Set to True if you have TLS enabled for MinIO
        )
        print(f"Successfully connected to MinIO at: {endpoint}")
        return minio_client
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        raise