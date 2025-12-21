"""
Test script to manually trigger video processing workflows.

This is useful for testing the workflow without going through the backend.

Usage:
    python trigger_workflow.py
"""

import asyncio
from temporalio.client import Client
from workflows import VideoProcessingWorkflow


async def main():
    """
    Trigger a test workflow for video processing.
    """
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    
    # Sample video data (replace with actual data)
    video_data = {
        "video_id": 999,  # Test video ID
        "file_name": "videos/94/test_video.mp4",  # Replace with actual S3 key
        "original_name": "test_video.mp4",
        "uploader_id": 1,
        "storage_path": "https://s3.example.com/videos/94/test_video.mp4",
    }
    
    print("🚀 Triggering video processing workflow...")
    print(f"Video ID: {video_data['video_id']}")
    print(f"File: {video_data['file_name']}")
    
    # Execute workflow
    result = await client.execute_workflow(
        VideoProcessingWorkflow.run,
        video_data,
        id=f"video-processing-{video_data['video_id']}",
        task_queue="video-processing-task-queue",
    )
    
    print("\n✅ Workflow completed!")
    print(f"Result: {result}")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
