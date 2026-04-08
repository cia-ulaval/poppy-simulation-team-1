import argparse
import asyncio
import logging
import queue
import sys
import threading
import time

import cv2
import numpy as np
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthClient:
    def __init__(
        self,
        host: str = "108.39.26.2",
        port: int = 46533,
        webcam_id: int = 0,
        target_fps: int = 15,
        jpeg_quality: int = 85,
        queue_size: int = 10,
    ):
        self.host = host
        self.port = port
        self.webcam_id = webcam_id
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.queue_size = queue_size

        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.cap = None
        self.ws = None

    def capture_thread_func(self):
        cap = cv2.VideoCapture(self.webcam_id)
        if not cap.isOpened():
            logger.error(f"Failed to open webcam {self.webcam_id}")
            return

        logger.info(f"Webcam {self.webcam_id} opened successfully")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                time.sleep(0.1)
                continue

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            _, encoded = cv2.imencode(".jpg", frame, encode_params)

            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(encoded.tobytes(), timeout=0.1)
            except queue.Full:
                pass

        cap.release()
        logger.info("Capture thread stopped")

    async def run(self):
        self.stop_event.clear()
        capture_thread = threading.Thread(target=self.capture_thread_func, daemon=True)
        capture_thread.start()

        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Connecting to {uri}")

        try:
            async with websockets.connect(uri) as websocket:
                logger.info("Connected to server")

                window_name = "Depth Estimation - Press 'q' to quit"
                frame_interval = 1.0 / self.target_fps

                while not self.stop_event.is_set():
                    start_time = time.time()

                    try:
                        frame_bytes = self.frame_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    try:
                        await websocket.send(frame_bytes)
                        depth_bytes = await websocket.recv()

                        nparr = np.frombuffer(depth_bytes, np.uint8)
                        depth_frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                        if depth_frame is not None:
                            cv2.imshow(window_name, depth_frame)

                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                logger.info("User requested quit")
                                self.stop_event.set()
                                break

                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Connection to server lost")
                        break
                    except Exception as e:
                        logger.error(f"Error communicating with server: {e}")
                        continue

                    elapsed = time.time() - start_time
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Client cancelled")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            self.stop_event.set()
            capture_thread.join(timeout=2.0)
            cv2.destroyAllWindows()
            logger.info("Client stopped")


def main():
    parser = argparse.ArgumentParser(description="Real-time depth estimation client")
    parser.add_argument(
        "--host",
        type=str,
        default="108.39.26.2",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=46533, help="Server port (default: 8765)"
    )
    parser.add_argument(
        "--webcam", type=int, default=0, help="Webcam device ID (default: 0)"
    )
    parser.add_argument(
        "--fps", type=int, default=15, help="Target display FPS (default: 15)"
    )
    parser.add_argument(
        "--quality", type=int, default=85, help="JPEG quality 1-100 (default: 85)"
    )
    parser.add_argument(
        "--queue-size", type=int, default=10, help="Frame queue size (default: 10)"
    )

    args = parser.parse_args()

    client = DepthClient(
        host=args.host,
        port=args.port,
        webcam_id=args.webcam,
        target_fps=args.fps,
        jpeg_quality=args.quality,
        queue_size=args.queue_size,
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
