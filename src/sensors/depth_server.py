import asyncio
import logging

import cv2
import numpy as np
import websockets
from PIL import Image
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"
PORT = 8000

depth_estimator = pipeline(
    task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
)


async def handle_client(websocket):
    logger.info("Client connected")
    try:
        async for message in websocket:
            try:
                nparr = np.frombuffer(message, np.uint8)
                frame_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame_np is None:
                    logger.warning("Failed to decode frame")
                    continue

                frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                outputs = depth_estimator(pil_image)

                depth_array = np.array(outputs["depth"])
                print(depth_array.shape)

                await websocket.send(depth_array.astype(np.float16).tobytes())

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


async def main():
    logger.info(f"Starting depth estimation server on {HOST}:{PORT}")
    logger.info("Depth model loaded and ready")

    async with websockets.serve(handle_client, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
