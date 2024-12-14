import mariadb
import os
from minio import Minio

def mariadb_connect() -> mariadb.Connection:
    """
    Establishes a connection to MariaDB using environment variables.
    """
    try:
        return mariadb.connect(
            user=os.environ.get("MARIADB_USER"),
            password=os.environ.get("MARIADB_PASSWORD"),
            host=os.environ.get("MARIADB_HOST"),
            port=int(os.environ.get("MARIADB_PORT", 3306)),  # Default port 3306
            database=os.environ.get("MARIADB_DATABASE")
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        raise  # Re-raise the exception to be handled by the caller

def minio_connect() -> Minio:
    """
    Establishes a connection to MinIO using environment variables.
    """
    minio_endpoint = os.environ.get("MINIO_ENDPOINT")
    minio_access_key = os.environ.get("MINIO_ACCESS_KEY")
    minio_secret_key = os.environ.get("MINIO_SECRET_KEY")
    # important?
    # minio_secure = os.environ.get("MINIO_SECURE", "False").lower() == "true"  # Default to False
    try:
        return Minio(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            #secure=minio_secure
        )
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        raise  # Re-raise the exception to be handled by the caller