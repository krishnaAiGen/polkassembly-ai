from fastapi import FastAPI, Response
import uvicorn
import os

app = FastAPI()

LOG_FILE = "logs/api_server.log"

@app.get("/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"error": "Log file not found."}

    with open(LOG_FILE, "r") as f:
        logs = f.read()

    return Response(content=logs, media_type="text/plain")

if __name__ == "__main__":
    # Run FastAPI app on host=0.0.0.0 and port=8000
    uvicorn.run("get_logs:app", host="0.0.0.0", port=8888, reload=True)