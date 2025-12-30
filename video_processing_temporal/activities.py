"""
Temporal Activities for Video Processing Pipeline

This module contains all the activities that will be executed by Temporal workers
for the video processing pipeline:
1. Download video from Backblaze S3
2. Process video with YOLO + InsightFace + DeepSORT
3. Store results in Qdrant vector database
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List
from temporalio import activity
from dataclasses import dataclass
import requests
import boto3
from botocore.client import Config
from dotenv import load_dotenv
import subprocess
import json
from datetime import datetime

BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://localhost:3000/internal/workflow-callback")
WORKER_CALLBACK_TOKEN = os.getenv("WORKER_CALLBACK_TOKEN", None)

def _post_status_update(video_id: int, stage: str, message: str = None, status: str = "info", error: str = None, details: dict = None):
    try:
        headers = {"Content-Type": "application/json"}
        if WORKER_CALLBACK_TOKEN:
            headers["x-worker-token"] = WORKER_CALLBACK_TOKEN
        payload = {
            "video_id": video_id,
            "stage": stage,
            "message": message,
            "status": status,
            "error": error,
            "workflow_id": getattr(activity.info, 'workflow_id', None),
            "workflow_run_id": getattr(activity.info, 'workflow_run_id', None),
            "details": details or {}
        }
        requests.post(BACKEND_INTERNAL_URL, json=payload, headers=headers, timeout=10)
    except Exception as e:
        activity.logger.warning(f"Callback post failed: {e}")

# Load environment variables from backend .env file
# Try multiple locations in case of different working directories
def load_env_config():
    """Load environment variables from .env file"""
    backend_env_paths = [
        # Absolute path from workspace root
        'C:/sabbas backend/VisionIndex-Backend/app/.env',
        # Relative from current script
        os.path.join(os.path.dirname(__file__), '../../VisionIndex-Backend/app/.env'),
        # Relative from current working directory
        os.path.join(os.getcwd(), '../VisionIndex-Backend/app/.env'),
        'VisionIndex-Backend/app/.env',
        '.env'  # Fallback to current directory
    ]
    
    for env_path in backend_env_paths:
        expanded_path = os.path.expanduser(os.path.expandvars(env_path))
        if os.path.exists(expanded_path):
            load_dotenv(expanded_path)
            logging.info(f"✅ Loaded environment from: {os.path.abspath(expanded_path)}")
            return True
    
    logging.warning("⚠️  No .env file found, using system environment variables")
    return False

load_env_config()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoProcessingInput:
    """Input data for video processing workflow"""
    video_id: int
    file_name: str  # S3 key (e.g., "videos/94/1764423168535_qmxes7itg2e.mp4")
    original_name: str
    uploader_id: int
    storage_path: str


@dataclass
class VideoProcessingResult:
    """Result of video processing"""
    video_id: int
    status: str
    total_persons: int
    total_objects: int
    processing_time_seconds: float
    error: str = None


# =============================================================================
# Utility Activity: Cleanup downloaded local files (video and reference image)
# =============================================================================

@activity.defn
async def cleanup_files_activity(paths: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete local files downloaded for processing to conserve disk space.

    Args:
        paths: Dict with optional keys 'video_path' and 'image_path'.

    Returns:
        Dict with deletion results per path.
    """
    results: Dict[str, Any] = {"video_deleted": False, "image_deleted": False}

    def _delete_file(path: str) -> bool:
        try:
            if path and os.path.isfile(path):
                os.remove(path)
                return True
            return False
        except Exception as e:
            activity.logger.warning(f"Failed to delete file {path}: {e}")
            return False

    video_path = (paths or {}).get("video_path")
    image_path = (paths or {}).get("image_path")

    if video_path:
        deleted = _delete_file(video_path)
        results["video_deleted"] = deleted
        activity.logger.info(f"Cleanup video: path={video_path} deleted={deleted}")

    if image_path:
        deleted = _delete_file(image_path)
        results["image_deleted"] = deleted
        activity.logger.info(f"Cleanup image: path={image_path} deleted={deleted}")

    return results


# =============================================================================
# Activity 1: Download Video from Backblaze S3
# =============================================================================

@activity.defn
async def download_video_activity(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download video from Backblaze S3 storage.
    
    Args:
        video_data: Dictionary containing video_id, file_name, etc.
    
    Returns:
        Dictionary with local_path to the downloaded video
    """
    activity.logger.info(f"Starting video download for video_id: {video_data['video_id']}")
    
    try:
        # Callback: stage started
        _post_status_update(video_data['video_id'], 'downloading', 'Download started')

        # S3 Configuration
        ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID")
        SECRET_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
        BUCKET = os.getenv("S3_BUCKET_NAME")
        ENDPOINT = os.getenv("S3_ENDPOINT")
        REGION = os.getenv("S3_REGION")
        
        # Debug logging
        activity.logger.info(f"S3 Config - Endpoint: {ENDPOINT}, Bucket: {BUCKET}, Region: {REGION}")
        if not ACCESS_KEY or ACCESS_KEY == "your_access_key_here":
            activity.logger.error(f"S3_ACCESS_KEY_ID is missing or invalid: {ACCESS_KEY}")
        if not SECRET_KEY or SECRET_KEY == "your_secret_key_here":
            activity.logger.error(f"S3_SECRET_ACCESS_KEY is missing or invalid: {SECRET_KEY}")
        
        if not all([ACCESS_KEY, SECRET_KEY, BUCKET, ENDPOINT, REGION]):
            raise ValueError(f"Missing S3 configuration. ACCESS_KEY: {bool(ACCESS_KEY)}, SECRET_KEY: {bool(SECRET_KEY)}, BUCKET: {bool(BUCKET)}, ENDPOINT: {bool(ENDPOINT)}, REGION: {bool(REGION)}")
        
        # Create S3 client
        s3 = boto3.client(
            "s3",
            region_name=REGION,
            endpoint_url=ENDPOINT,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
        
        file_name = video_data['file_name']
        
        # Generate presigned URL
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": file_name},
            ExpiresIn=3600,
        )
        
        # Ensure output folder exists
        output_folder = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "download_video", 
            "downloaded_videos"
        )
        os.makedirs(output_folder, exist_ok=True)
        
        # Output path
        local_filename = os.path.basename(file_name)
        local_path = os.path.join(output_folder, local_filename)
        
        activity.logger.info(f"Downloading from: {ENDPOINT}/{BUCKET}/{file_name}")
        activity.logger.info(f"Saving to: {local_path}")
        
        # Stream download with heartbeat
        response = requests.get(presigned_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Send heartbeat every MB to show progress
                    if downloaded_size % (1024 * 1024) == 0:  # Every 1 MB
                        percent = (downloaded_size / total_size * 100) if total_size > 0 else 0
                        activity.logger.info(f"Download progress: {percent:.1f}% ({downloaded_size}/{total_size} bytes)")
                        activity.heartbeat({"progress": percent, "bytes_downloaded": downloaded_size})
        
        activity.logger.info(f"[DOWNLOAD-COMPLETE] Downloaded {downloaded_size} bytes to {local_path}")
        _post_status_update(video_data['video_id'], 'preprocessing', 'Download complete', details={"bytes": downloaded_size, "path": local_path})
        
        return {
            "video_id": video_data['video_id'],
            "local_path": local_path,
            "file_size": downloaded_size,
            "status": "downloaded"
        }
        
    except Exception as e:
        activity.logger.error(f"❌ Error downloading video: {e}")
        _post_status_update(video_data.get('video_id', -1), 'failed', 'Download failed', status='error', error=str(e))
        raise


# =============================================================================
# Activity 1b: Download Reference Image from S3 (Optional)
# =============================================================================

@activity.defn
async def download_reference_image_activity(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download reference image from S3 for post-processing face comparison.
    If image_data is empty or None, this is a no-op and returns a flag.
    
    Args:
        image_data: Dictionary containing:
            - image_id: int
            - file_name: str (S3 key, e.g., "images/94/timestamp-xxxxx.jpg")
            - video_id: int (for logging)
    
    Returns:
        Dictionary with local_path to the downloaded image, or status='no_image' if None
    """
    if not image_data:
        activity.logger.info("No reference image provided; skipping download")
        return {"status": "no_image", "local_path": None}
    
    video_id = image_data.get('video_id')
    image_id = image_data.get('image_id')
    
    activity.logger.info(f"Starting reference image download for video_id: {video_id}, image_id: {image_id}")
    
    try:
        # S3 Configuration (reuse same credentials as video)
        ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID")
        SECRET_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
        BUCKET = os.getenv("S3_BUCKET_NAME")
        ENDPOINT = os.getenv("S3_ENDPOINT")
        REGION = os.getenv("S3_REGION")
        
        if not all([ACCESS_KEY, SECRET_KEY, BUCKET, ENDPOINT, REGION]):
            raise ValueError("Missing S3 configuration for image download")
        
        # Create S3 client
        s3 = boto3.client(
            "s3",
            region_name=REGION,
            endpoint_url=ENDPOINT,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
        
        file_name = image_data['file_name']
        
        # Generate presigned URL
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": file_name},
            ExpiresIn=3600,
        )
        
        # Ensure output folder exists
        output_folder = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "download_video", 
            "reference_images"
        )
        os.makedirs(output_folder, exist_ok=True)
        
        # Output path
        local_filename = os.path.basename(file_name)
        local_path = os.path.join(output_folder, local_filename)
        
        activity.logger.info(f"Downloading reference image from: {ENDPOINT}/{BUCKET}/{file_name}")
        activity.logger.info(f"Saving to: {local_path}")
        
        # Stream download
        response = requests.get(presigned_url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
        
        activity.logger.info(f"✅ Reference image downloaded: {downloaded_size} bytes to {local_path}")
        
        return {
            "status": "downloaded",
            "image_id": image_id,
            "video_id": video_id,
            "local_path": local_path,
            "file_size": downloaded_size
        }
        
    except Exception as e:
        activity.logger.error(f"❌ Error downloading reference image: {e}")
        # Don't fail the workflow; just return no_image and processing will use default
        return {"status": "error", "local_path": None, "error": str(e)}


# =============================================================================
# Activity 2: Process Video with Pipeline
# =============================================================================

@activity.defn
async def process_video_activity(process_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process video using the cursor_muk_deepsort.py pipeline.
    
    Args:
        process_input: Dictionary containing:
            - download_result: Result from download_video_activity
            - image_result: Result from download_reference_image_activity (optional)
            - text_query: Text query for search (optional)
    
    Returns:
        Dictionary with processing results
    """
    download_result = process_input.get('download_result')
    image_result = process_input.get('image_result')
    text_query = process_input.get('text_query')
    
    if not download_result:
        raise ValueError("Missing download_result in process_input")
    
    # Either image or text query must be provided
    if not image_result and not text_query:
        raise ValueError("Either reference image or text query must be provided")
    
    video_id = download_result['video_id']
    local_path = download_result['local_path']
    
    activity.logger.info(f"[PROCESS-START] Starting video processing for video_id: {video_id}")
    activity.logger.info(f"[PROCESS-FILE] Processing file: {local_path}")
    
    try:
        # Callback: processing start
        _post_status_update(video_id, 'analysis', 'Processing started', details={"file": local_path})

        # Verify file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Video file not found: {local_path}")
        
        # Get the pipeline script path
        pipeline_dir = os.path.join(os.path.dirname(__file__), "..", "pipeline")
        pipeline_script = os.path.join(pipeline_dir, "cursor_muk_deepsort.py")
        
        if not os.path.exists(pipeline_script):
            raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")
        
        # Prepare environment variables for the pipeline
        env = os.environ.copy()
        env['VIDEO_ID'] = str(video_id)
        env['VIDEO_PATH'] = local_path
        
        # Set reference image if available; otherwise ensure any leaked env is removed
        if image_result and image_result.get('status') == 'downloaded' and image_result.get('local_path'):
            env['REFERENCE_FACE_IMAGE'] = image_result['local_path']
            activity.logger.info(f"[PROCESS-IMAGE] Using reference image: {image_result['local_path']}")
        else:
            # Remove any pre-existing REFERENCE_FACE_IMAGE from parent env to prevent accidental default usage
            if 'REFERENCE_FACE_IMAGE' in env:
                env.pop('REFERENCE_FACE_IMAGE', None)
                activity.logger.info("[PROCESS-IMAGE] No reference image provided; cleared REFERENCE_FACE_IMAGE from env")
        
        # Set text query if available
        if text_query:
            env['TEXT_QUERY'] = text_query
            activity.logger.info(f"[PROCESS-TEXT] Using text query: {text_query}")
        
        # Run the pipeline as a subprocess
        activity.logger.info(f"Executing pipeline: python {pipeline_script}")
        
        # Run pipeline with timeout and capture output
        process = await asyncio.create_subprocess_exec(
            sys.executable,  # Use current Python interpreter
            pipeline_script,
            local_path,  # Pass video path as argument
            str(video_id),  # Pass video_id as argument
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=pipeline_dir
        )
        
        activity.logger.info(f"[PIPELINE-RUNNING] Pipeline process started with PID: {process.pid}")
        
        # Monitor process with heartbeats
        heartbeat_task = asyncio.create_task(_send_heartbeats_during_processing(process))
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=3600  # 1 hour timeout
            )
        except asyncio.CancelledError:
            activity.logger.error(f"Processing cancelled (timeout or workflow cancellation) for video_id: {video_id}")
            _post_status_update(video_id, 'failed', 'Processing cancelled', status='error')
            try:
                process.kill()
                await asyncio.wait_for(process.communicate(), timeout=5)
            except Exception as kill_err:
                activity.logger.warning(f"Failed to cleanly kill pipeline process: {kill_err}")
            raise
        finally:
            heartbeat_task.cancel()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            activity.logger.error(f"Pipeline failed with return code {process.returncode}")
            activity.logger.error(f"Error: {error_msg}")
            raise RuntimeError(f"Pipeline processing failed: {error_msg}")
        
        # Parse output for statistics and query results
        output = stdout.decode()
        activity.logger.info("Pipeline output (last 500 chars):")
        activity.logger.info(output[-500:])
        
        # Extract statistics from output if available
        total_persons = 0
        total_objects = 0
        query_results = []
        
        # Try to extract JSON results from output
        try:
            # Look for [RESULTS-JSON] marker and extract JSON
            if "[RESULTS-JSON]" in output:
                json_start = output.find("[RESULTS-JSON]") + len("[RESULTS-JSON]")
                json_end = output.find("="*80, json_start)
                if json_end > json_start:
                    json_str = output[json_start:json_end].strip()
                    results_data = json.loads(json_str)
                    query_results = results_data.get("query_results", [])
                    activity.logger.info(f"✅ Extracted {len(query_results)} query results from pipeline")
        except json.JSONDecodeError as e:
            activity.logger.warning(f"Could not parse JSON results from pipeline output: {e}")
        except Exception as e:
            activity.logger.warning(f"Error extracting query results: {e}")
        
        activity.logger.info(f"✅ Video processing complete for video_id: {video_id}")
        _post_status_update(video_id, 'postprocessing', 'Processing complete')
        
        return {
            "video_id": video_id,
            "status": "processed",
            "total_persons": total_persons,
            "total_objects": total_objects,
            "local_path": local_path,
            "query_results": query_results
        }
        
    except asyncio.TimeoutError:
        activity.logger.error(f"❌ Processing timeout for video_id: {video_id}")
        _post_status_update(video_id, 'failed', 'Processing timeout', status='error')
        raise RuntimeError("Video processing timeout (exceeded 1 hour)")
    
    except Exception as e:
        activity.logger.error(f"❌ Error processing video: {e}")
        _post_status_update(video_id, 'failed', 'Processing failed', status='error', error=str(e))
        raise


async def _send_heartbeats_during_processing(process):
    """Send heartbeats while processing is ongoing"""
    while process.returncode is None:
        try:
            activity.heartbeat("Processing video...")
            await asyncio.sleep(10)  # Heartbeat every 10 seconds
        except asyncio.CancelledError:
            break


# =============================================================================
# Activity 3: Store Results in Qdrant
# =============================================================================

@activity.defn
async def store_results_activity(processing_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that results were stored in Qdrant during processing.
    The pipeline (cursor_muk_deepsort.py) already stores results in Qdrant,
    so this activity mainly verifies the storage and updates status.
    
    Args:
        processing_result: Result from process_video_activity (includes query_results if text/image query was run)
    
    Returns:
        Dictionary with final results including query_results if applicable
    """
    video_id = processing_result['video_id']
    query_results = processing_result.get('query_results', [])
    
    activity.logger.info(f"Verifying Qdrant storage for video_id: {video_id}")
    
    try:
        from qdrant_client import QdrantClient
        from urllib.parse import urlparse
        
        # Connect to Qdrant
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6334")
        
        client = None
        if qdrant_url:
            parsed = urlparse(qdrant_url)
            if parsed.port == 6334:
                client = QdrantClient(host=parsed.hostname, port=parsed.port, prefer_grpc=True)
            else:
                client = QdrantClient(url=qdrant_url)
        else:
            client = QdrantClient(host=qdrant_host, port=int(qdrant_port), prefer_grpc=True)
        
        # Verify collections exist
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        activity.logger.info(f"Available Qdrant collections: {collection_names}")
        
        # Check if person_tracks and object_tracks exist
        if "person_tracks" not in collection_names or "object_tracks" not in collection_names:
            raise RuntimeError("Required Qdrant collections not found")
        
        # Count records for this video_id
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        person_count = client.count(
            collection_name="person_tracks",
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )
        )
        
        object_count = client.count(
            collection_name="object_tracks",
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )
        )
        
        activity.logger.info(f"Found {person_count.count} person tracks and {object_count.count} object tracks")
        
        # Clean up downloaded video file to save space
        local_path = processing_result.get('local_path')
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                activity.logger.info(f"Cleaned up video file: {local_path}")
            except Exception as e:
                activity.logger.warning(f"Could not delete video file: {e}")
        
        activity.logger.info(f"✅ Results verified and stored for video_id: {video_id}")
        _post_status_update(video_id, 'completed', 'Results stored', details={"persons": person_count.count, "objects": object_count.count})
        
        return {
            "video_id": video_id,
            "status": "completed",
            "total_persons": person_count.count,
            "total_objects": object_count.count,
            "qdrant_verified": True,
            "query_results": query_results  # Pass through any query results
        }
        
    except Exception as e:
        activity.logger.error(f"❌ Error verifying Qdrant storage: {e}")
        _post_status_update(video_id, 'completed', 'Storage verification failed', status='warning', error=str(e))
        # Don't fail the workflow if verification fails, just log it
        return {
            "video_id": video_id,
            "status": "completed_with_warnings",
            "total_persons": 0,
            "total_objects": 0,
            "qdrant_verified": False,
            "query_results": query_results,
            "error": str(e)
        }


# =============================================================================
# Activity 4: Update Database Status (Optional)
# =============================================================================

@activity.defn
async def update_video_status_activity(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update video status in PostgreSQL database.
    This can be called from the backend instead, but having it here
    provides more flexibility.
    
    Args:
        result: Final processing result
    
    Returns:
        Updated result
    """
    video_id = result['video_id']
    status = result['status']
    
    activity.logger.info(f"Updating database status for video_id: {video_id} to {status}")
    
    try:
        # This would typically use psycopg2 or asyncpg to update the database
        # For now, we'll just log it and let the backend handle it via polling
        # or by listening to workflow events
        
        activity.logger.info(f"✅ Status update logged for video_id: {video_id}")
        
        return result
        
    except Exception as e:
        activity.logger.error(f"❌ Error updating database: {e}")
        # Don't fail workflow if status update fails
        return result
