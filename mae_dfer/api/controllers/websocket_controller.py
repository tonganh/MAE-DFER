import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mae_dfer.api.services import config, inference_service, model_registry


router = APIRouter()


@router.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    if not model_registry.model_loaded():
        await websocket.close(code=1011)
        return
    segment_index = 0
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"error": "invalid_json", "segment_index": segment_index}
                    )
                    continue
                if payload.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                continue
            if msg.get("bytes") is not None:
                data = msg["bytes"]
                name = f"segment_{segment_index}.bin"
                try:
                    out = await asyncio.to_thread(inference_service.predict_bytes_sync, data, name)
                    out["segment_index"] = segment_index
                    out["segment_duration_hint_sec"] = config.stream_segment_sec()
                    await websocket.send_json(out)
                    segment_index += 1
                except ValueError as e:
                    await websocket.send_json(
                        {
                            "error": str(e),
                            "segment_index": segment_index,
                        }
                    )
    except WebSocketDisconnect:
        pass

