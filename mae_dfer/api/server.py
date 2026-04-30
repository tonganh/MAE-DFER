import uvicorn
import os

from mae_dfer.api.app import create_app


app = create_app()


def main():
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    ws_max = int(os.environ.get("WS_MAX_SIZE", str(50 * 1024 * 1024)))
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=1,
        ws_max_size=ws_max,
    )


if __name__ == "__main__":
    main()
