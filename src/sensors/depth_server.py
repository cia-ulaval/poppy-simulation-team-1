import asyncio
import logging
import traceback

import cv2
import numpy as np
import torch
import websockets
from depth_anything_3.api import DepthAnything3
from PIL import Image
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"
PORT = 8000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
model = model.to(device=device)


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

                print("Decoded frame")
                frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image.save("input.png")

                print("Running inference")

                outputs = model.inference(
                    "input.png",
                    export_format="npz",  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
                )
                print("Inference completed")

                depth_array = np.array(outputs["depth"][0])
                print(depth_array.shape)

                await websocket.send(depth_array.astype(np.float16).tobytes())

            except Exception as e:
                logger.error(f"Error processing frame: {traceback.format_exc()}")
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
