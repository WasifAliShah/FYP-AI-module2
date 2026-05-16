"""
Temporal Worker for Video Processing

This worker runs the video processing activities and workflows.
It should be started before uploading videos.

To run:
    python worker.py

Make sure Temporal server is running:
    docker run -p 7233:7233 temporalio/auto-setup:latest
"""

import asyncio
import logging
import sys

# Debug: Print starting message immediately
print("[START] Worker starting...", file=sys.stderr, flush=True)
print("[START] Worker starting...", flush=True)

from temporalio.client import Client
from temporalio.worker import Worker

from workflows import (
    VideoProcessingWorkflow,
    BatchVideoProcessingWorkflow,
    VideoReprocessingWorkflow,
)
from activities import (
    download_video_activity,
    download_reference_image_activity,
    process_video_activity,
    store_results_activity,
    update_video_status_activity,
    cleanup_files_activity,
)

print("[OK] All imports successful", flush=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('worker.log')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """
    Main function to start the Temporal worker.
    """
    print("[MAIN] Starting main async function", flush=True)
    
    import os
    # Connect to Temporal server
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    
    logger.info(f"Connecting to Temporal server at {temporal_host}...")
    print(f"[CONNECT] Connecting to Temporal at {temporal_host}", flush=True)
    
    try:
        client = await Client.connect(temporal_host)
        logger.info("✅ Connected to Temporal server")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Temporal server: {e}")
        logger.error("Make sure Temporal server is running:")
        logger.error("  docker run -p 7233:7233 temporalio/auto-setup:latest")
        return
    
    # Create worker
    logger.info("Starting Temporal worker...")
    
    worker = Worker(
        client,
        task_queue="video-processing-task-queue",
        workflows=[
            VideoProcessingWorkflow,
            BatchVideoProcessingWorkflow,
            VideoReprocessingWorkflow,
        ],
        activities=[
            download_video_activity,
            download_reference_image_activity,
            process_video_activity,
            store_results_activity,
            update_video_status_activity,
            cleanup_files_activity,
        ],
        max_concurrent_activities=3,  # Process up to 3 videos concurrently
        max_concurrent_workflow_tasks=10,
    )
    
    logger.info("="*80)
    logger.info("🎬 Video Processing Worker Started!")
    logger.info("="*80)
    logger.info("Task Queue: video-processing-task-queue")
    logger.info("Max Concurrent Activities: 3")
    logger.info("")
    logger.info("Available Workflows:")
    logger.info("  - VideoProcessingWorkflow")
    logger.info("  - BatchVideoProcessingWorkflow")
    logger.info("  - VideoReprocessingWorkflow")
    logger.info("")
    logger.info("Available Activities:")
    logger.info("  - download_video_activity")
    logger.info("  - download_reference_image_activity")
    logger.info("  - process_video_activity")
    logger.info("  - store_results_activity")
    logger.info("  - update_video_status_activity")
    logger.info("  - cleanup_files_activity")
    logger.info("")
    logger.info("Waiting for workflow tasks...")
    logger.info("="*80)
    
    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("\n🛑 Worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Worker error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
