# Frontend implementation guide (MAE-DFER API)

This document describes how the **demo** client should talk to the backend (`api_server.py`). The demo uses **HTTP only** (no WebSocket).

## Demo flow (what to build)

1. User records **about 10 seconds** of video in the browser (camera or file).
2. Client sends that file to the backend with **`POST /predict`** (multipart field `file`).
3. Backend runs the model and returns **JSON**.
4. Frontend **displays** the prediction (label, probabilities, optional chart).

Repeat steps 1–4 for another clip if you want a new prediction; each request is independent.

---

## Base URL

- Local example: `http://127.0.0.1:8000`
- Use your real host/port in production (`API_HOST`, `API_PORT` on the server).

## CORS

If the web app and API are on **different origins**, configure CORS on the backend or proxy `/predict` through the same origin as the frontend.

---

## Health check (optional)

**`GET /health`**

```json
{
  "status": "ok",
  "device": "cuda:0",
  "checkpoint": "/path/to/checkpoint.pth"
}
```

---

## Prediction (demo): `POST /predict`

**Request**

- Method: `POST`
- Content type: `multipart/form-data`
- Field name: **`file`** (required)

| Field  | Type | Description        |
|--------|------|--------------------|
| `file` | file | Video file         |

**Allowed filename extensions:** `.mp4`, `.avi`, `.mov`, `.webm`, `.mkv`

**Rules**

- File must be at least **1024 bytes** (otherwise `400`).
- Unsupported extension → `400`.

**Success response (JSON)**

```json
{
  "predicted_class_index": 0,
  "predicted_label": "Happy",
  "class_names": ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"],
  "probabilities": {
    "Happy": 0.82,
    "Sad": 0.02,
    "Neutral": 0.05,
    "Angry": 0.03,
    "Surprise": 0.04,
    "Disgust": 0.02,
    "Fear": 0.02
  },
  "filename": "clip.mp4"
}
```

Use **`predicted_label`** and **`probabilities`** for the UI. Map **`probabilities`** to a bar list or similar.

**Error**

- HTTP `4xx` / `5xx` with JSON `detail` (FastAPI) or message body as returned by the server.

---

## Browser demo: record ~10s, then `POST`

Use **`MediaRecorder`** with a **10 000 ms** timeslice so each blob is roughly one 10 second clip, **or** stop the recorder after 10 seconds—either way you get a **`Blob`** to upload.

```javascript
const API_BASE = "http://127.0.0.1:8000";

async function predictBlob(videoBlob, filename = "clip.webm") {
  const form = new FormData();
  form.append("file", videoBlob, filename);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail ?? `${res.status} ${res.statusText}`);
  }
  return data;
}

function displayResult(data) {
  const label = data.predicted_label;
  const probs = data.probabilities;
  document.getElementById("label").textContent = label;
  console.table(probs);
}

async function runDemo() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
    ? "video/webm;codecs=vp8"
    : "video/webm";
  const recorder = new MediaRecorder(stream, { mimeType: mime });

  recorder.ondataavailable = async (ev) => {
    if (!ev.data || ev.data.size < 1024) return;
    stream.getTracks().forEach((t) => t.stop());
    try {
      const data = await predictBlob(ev.data, "segment.webm");
      displayResult(data);
    } catch (e) {
      console.error(e);
    }
  };

  recorder.start(10000);
}
```

Notes:

- Pick a `mimeType` the browser supports (`MediaRecorder.isTypeSupported`).
- After the first `ondataavailable` (10 s), the example stops tracks so the camera turns off; adjust if you want repeated 10 s rounds (call `predictBlob` each time without stopping, or restart recording).
- For a **file upload** demo (user picks a `.mp4`), use `input type="file"` and pass `file` from the input into `FormData` the same way.

---

## Mobile / native

Same as the demo: **`POST /predict`** with multipart **`file`**. No WebSocket required.

---

## Optional: WebSocket (not part of the demo)

The server may expose **`/ws/predict`** for long-lived sessions with multiple binary segments. **Demo frontends do not need it**; use **`POST /predict`** only.
