import logging
import uuid
import multiprocessing as mp

from fastapi import FastAPI, HTTPException, status
from schemas import *
from database import mariadb_connect
from worker import Worker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

request_queue = mp.Queue()
worker = Worker(request_queue)

@app.post("/inference/", response_model=InferenceResponse)
async def inference(request: ClientRequest):
    # create unique identifier
    id = uuid.uuid4()
    logger.info(f"Received inference request with ID: {id}")

    try:
        with mariadb_connect() as conn:
            cursor = conn.cursor()

            cursor.execute(f"SELECT * FROM requests WHERE path_to_model = '{request.path_to_data}'")
            existing_request = cursor.fetchone()
            logger.info(f"Checked for existing requests for path: {request.path_to_data}")

            if existing_request:
                request_id, status, *_ = existing_request  # Unpack the tuple
                logger.info(f"Found existing request with ID: {request_id}, Status: {status}")
                if status == "RUNNING":
                    logger.warning(f"Request with ID: {request_id} is already running.")
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Request already running")
                else:
                    cursor.execute("DELETE FROM requests WHERE id = %s", (request_id,))
                    conn.commit()
                    logger.info(f"Deleted old request with ID: {request_id}")

            cursor.execute("insert into requests (uuid, status, path_to_model) values (%s, %s, %s)", (str(id), "WAITING", request.path_to_data))
            conn.commit()
            logger.info(f"Inserted new request with ID: {id}, Status: WAITING")

            req = InferenceRequest(path_to_model=request.path_to_data, uuid=str(id))
            request_queue.put(req)
            logger.info(f"Put request {req} into queue.")

        # return uuid in response
        return InferenceResponse(id=str(id))

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")