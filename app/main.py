import uuid
import multiprocessing as mp

from fastapi import FastAPI, HTTPException, status
from schemas import *
from database import mariadb_connect, create_requests_table
from worker import Worker


app = FastAPI()

request_queue = mp.Queue()
worker = Worker(request_queue)
worker.start()

create_requests_table()

@app.post("/inference/", response_model=InferenceResponse)
async def inference(request: ClientRequest):
    # create unique identifier
    id = uuid.uuid4()
    print(f"Received inference request with ID: {id}")

    try:
        with mariadb_connect() as conn:
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM requests WHERE path_to_model = '{request.path_to_data}'")
            existing_request = cursor.fetchone()
            print(f"Checked for existing requests for path: {request.path_to_data}")

            if existing_request:
                request_id, status, *_ = existing_request  # Unpack the tuple
                print(f"Found existing request with ID: {request_id}, Status: {status}")
                if status == "RUNNING":
                    print(f"Request with ID: {request_id} is already running.")
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Request already running")
                else:
                    cursor.execute("DELETE FROM requests WHERE uuid = %s", (request_id,))
                    conn.commit()
                    print(f"Deleted old request with ID: {request_id}")

            cursor.execute("insert into requests (uuid, status, path_to_model) values (%s, %s, %s)", (str(id), "WAITING", request.path_to_data))
            conn.commit()
            print(f"Inserted new request with ID: {id}, Status: WAITING")

            req = InferenceRequest(path_to_model=request.path_to_data, uuid=str(id))
            request_queue.put(req)
            print(f"Put request {req} into queue.")

        # return uuid in response
        return InferenceResponse(id=str(id))

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")