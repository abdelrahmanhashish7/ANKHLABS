# proxy_app.py
import asyncio
import time
from typing import List, Optional
from fastapi import FastAPI, Request
import httpx
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# Configuration
UPSTREAM_URL = "http://127.0.0.1:8000/data"  # change to your Flask /data if remote
FLUSH_MS = int(500)           # flush every 500 ms
MAX_BATCH = 500               # flush if buffer reaches this many samples
ACK_IMMEDIATE = True          # immediately ack ESP (true)

# In-memory buffers per device_id
buffers = {}         # device_id -> list of samples
buffers_lock = asyncio.Lock()

class DataPayload(BaseModel):
    device_id: Optional[str] = "default"
    ecg: Optional[List[float]] = None
    glucose: Optional[float] = None
    timestamp: Optional[float] = None

@app.on_event("startup")
async def startup_flush_task():
    asyncio.create_task(flush_loop())

@app.post("/data")
async def receive_data(payload: DataPayload, request: Request):
    """
    Accepts JSON:
    { "device_id": "esp1", "ecg": [..samples..] }
    or { "device_id":"esp1", "ecg": 123 }
    """
    device = payload.device_id or "default"

    # normalize ecg to a list
    samples = []
    if payload.ecg is None:
        # optional: maybe only glucose or other
        samples = []
    elif isinstance(payload.ecg, list):
        samples = payload.ecg
    else:
        samples = [payload.ecg]

    async with buffers_lock:
        if device not in buffers:
            buffers[device] = []
        # append (value, timestamp) pairs (server time)
        ts = payload.timestamp or time.time()
        buffers[device].extend([{"ecg": v, "timestamp": ts} for v in samples])

        # if buffer too large, schedule immediate flush by returning a header (flush will run)
        if len(buffers[device]) >= MAX_BATCH:
            # let flush_loop pick it up; we respond immediately
            pass

    # immediate lightweight ack so ESP doesn't block
    return {"status": "received", "queued": len(buffers.get(device, []))}


async def flush_loop():
    client = httpx.AsyncClient(timeout=10.0)
    while True:
        await asyncio.sleep(FLUSH_MS / 1000.0)
        async with buffers_lock:
            to_send = {}
            for device, arr in list(buffers.items()):
                if not arr:
                    continue
                # criteria to flush:
                if len(arr) >= MAX_BATCH or True:
                    # take all current items to send
                    to_send[device] = arr.copy()
                    buffers[device] = []
        # send outside lock
        for device, arr in to_send.items():
            # prepare upstream payload format expected by Flask
            # We'll send {"device_id": device, "ecg": [v1, v2, ...], "timestamp": now}
            try:
                ecg_list = [entry["ecg"] for entry in arr]
                payload = {"device_id": device, "ecg": ecg_list, "timestamp": time.time()}
                # If your Flask expects a different format, adapt here
                r = await client.post(UPSTREAM_URL, json=payload)
                if r.status_code != 200:
                    print("[proxy] upstream error:", r.status_code, r.text)
                    # On failure: requeue items for next try
                    async with buffers_lock:
                        buffers.setdefault(device, [])
                        # prepend failed items so we keep order
                        buffers[device] = arr + buffers[device]
                else:
                    # success
                    pass
            except Exception as e:
                print("[proxy] exception posting upstream:", e)
                # requeue
                async with buffers_lock:
                    buffers.setdefault(device, [])
                    buffers[device] = arr + buffers[device]

    # never reach client.aclose() in normal loop


if __name__ == "__main__":
    uvicorn.run("proxy_app:app", host="0.0.0.0", port=8080, workers=1)

