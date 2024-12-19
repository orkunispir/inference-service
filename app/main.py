import uuid
import multiprocessing as mp

from fastapi import FastAPI, HTTPException, status
from schemas import *
from database import mariadb_connect, create_requests_table
from worker import Worker


app = FastAPI()

create_requests_table()

print("Called create table function")

request_queue = mp.Queue()
worker = Worker(request_queue)
worker.start()

@app.post("/train/", response_model=TrainingResponse)
async def inference(request: ClientRequest):
    # create unique identifier
    id = uuid.uuid4()
    print(f"Received inference request with ID: {id}")

    try:
        with mariadb_connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM requests WHERE bucket = %s AND object_name = %s",
                (request.bucket, request.object_name),
            )
            existing_request = cursor.fetchone()
            print(
                f"Checked for existing requests for bucket: {request.bucket}, object_name: {request.object_name}"
            )

            if existing_request:
                request_id, status, *_ = existing_request  # Unpack the tuple
                print(f"Found existing request with ID: {request_id}, Status: {status}")
                if status == "TRAINING":
                    print(f"Request with ID: {request_id} is already running.")
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Request already running")
                else:
                    cursor.execute("DELETE FROM requests WHERE uuid = %s", (request_id,))
                    conn.commit()
                    print(f"Deleted old request with ID: {request_id}")

            cursor.execute(
                "INSERT INTO requests (uuid, status, bucket, object_name, model_bucket, model_object_name, created_at, updated_at) VALUES (%s, %s, %s, %s, NULL, NULL, NOW(), NOW())",
                (str(id), "WAITING", request.bucket, request.object_name),
            )
            conn.commit()
            print(f"Inserted new request with ID: {id}, Status: WAITING")

            req = TrainingRequest(
                bucket=request.bucket, object_name=request.object_name, uuid=str(id)
            )
            request_queue.put(req)
            print(f"Put request {req} into queue.")

        # return uuid in response
        return TrainingResponse(id=str(id))

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
@app.get("/status/{request_id}", response_model=StatusResponse)
async def get_status(request_id: str):
    """
    Retrieves the status of a training request by its UUID.
    """
    try:
        with mariadb_connect() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT status, model_bucket, model_object_name FROM requests WHERE uuid = %s", (request_id,))
            result = cursor.fetchone()

            if result:
                status, model_bucket, model_object_name = result
                print(f"Retrieved status for request ID {request_id}: Status={status}, Model Bucket={model_bucket}, Model Object Name={model_object_name}")
                return StatusResponse(status=status, model_bucket=model_bucket, model_object_name=model_object_name)
            else:
                print(f"No request found with ID: {request_id}")
                raise HTTPException(status_code=404, detail="Request not found")

    except Exception as e:
        print(f"An error occurred while retrieving status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")