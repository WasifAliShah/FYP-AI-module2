"""
Temporal Workflows for Video Processing Pipeline

This module defines the main workflow that orchestrates the video processing:
1. Download video from S3
2. Process video with ML pipeline
3. Store results in Qdrant
4. Update database status
"""

from datetime import timedelta
from typing import Dict, Any
from temporalio import workflow
from temporalio.common import RetryPolicy

# Import activity types
with workflow.unsafe.imports_passed_through():
    from activities import (
        download_video_activity,
        download_reference_image_activity,
        process_video_activity,
        store_results_activity,
        update_video_status_activity,
    )
    # Import module to access symbols not reliably imported by name in sandbox
    import activities as activities_module


@workflow.defn
class VideoProcessingWorkflow:
    """
    Main workflow for processing uploaded videos.
    
    This workflow:
    1. Downloads the video from Backblaze S3
    2. Processes it with YOLO + InsightFace + DeepSORT pipeline
    3. Stores embeddings and results in Qdrant
    4. Updates the video status in the database
    
    The workflow is designed to be resilient with retries and proper error handling.
    """
    
    @workflow.run
    async def run(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete video processing pipeline.
        
        Args:
            video_data: Dictionary containing:
                - video_id: int
                - file_name: str (S3 key)
                - original_name: str
                - uploader_id: int
                - storage_path: str
        
        Returns:
            Dictionary with final processing results
        """
        workflow.logger.info(f"🚀 Starting video processing workflow for video_id: {video_data['video_id']}")
        
        video_id = video_data['video_id']
        
        # Define retry policy for activities
        # This will retry on failures with exponential backoff
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            maximum_attempts=3,
        )
        
        # Track paths for cleanup in both success and failure cases
        download_result = None
        image_result = None

        try:
            # =========================================================================
            # Step 1: Download Video from S3
            # =========================================================================
            workflow.logger.info(f"📥 Step 1: Downloading video {video_data['file_name']}")
            
            download_result = await workflow.execute_activity(
                download_video_activity,
                video_data,
                start_to_close_timeout=timedelta(minutes=30),  # 30 min for large videos
                retry_policy=retry_policy,
            )
            
            workflow.logger.info(f"✅ Download complete: {download_result['local_path']}")
            
            # =========================================================================
            # Step 1b: Download Reference Image from S3 (if associated with video)
            # =========================================================================
            workflow.logger.info(f"📥 Step 1b: Checking for associated reference image")
            
            # Note: In a real scenario, you would query the database here to get image_data
            # For now, this is optional and can be passed in video_data if available
            image_data = video_data.get('image_data', None)
            
            image_result = None
            if image_data:
                image_result = await workflow.execute_activity(
                    download_reference_image_activity,
                    image_data,
                    start_to_close_timeout=timedelta(minutes=5),  # 5 min for image download
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=30),
                        backoff_coefficient=2.0,
                        maximum_attempts=2,
                    ),
                )
                
                if image_result.get('status') == 'downloaded':
                    workflow.logger.info(f"✅ Reference image downloaded: {image_result['local_path']}")
                else:
                    workflow.logger.info(f"⚠ No reference image to download (status: {image_result.get('status')})")
            else:
                workflow.logger.info(f"⚠ No reference image data provided")
            
            # =========================================================================
            # Step 2: Process Video with ML Pipeline
            # =========================================================================
            workflow.logger.info(f"🎬 Step 2: Processing video with ML pipeline")
            
            # Either image or text query must be provided
            has_image = image_result and image_result.get('status') == 'downloaded'
            text_query = video_data.get('text_query', None)
            
            if not has_image and not text_query:
                workflow.logger.error(f"⚠ Neither image nor text query provided")
                raise RuntimeError("Either reference image or text query must be provided")
            
            # Combine download and results into a single argument dict
            process_input = {
                "download_result": download_result,
                "image_result": image_result if has_image else None,
                "text_query": text_query
            }
            
            processing_result = await workflow.execute_activity(
                process_video_activity,
                process_input,
                start_to_close_timeout=timedelta(hours=2),  # 2 hours for processing
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=5),
                    maximum_interval=timedelta(minutes=5),
                    backoff_coefficient=2.0,
                    maximum_attempts=2,  # Only retry once for processing
                ),
            )
            
            workflow.logger.info(f"✅ Processing complete")
            
            # =========================================================================
            # Step 3: Verify and Store Results in Qdrant
            # =========================================================================
            workflow.logger.info(f"💾 Step 3: Verifying Qdrant storage")
            
            storage_result = await workflow.execute_activity(
                store_results_activity,
                processing_result,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=retry_policy,
            )
            
            workflow.logger.info(
                f"✅ Storage verified - Persons: {storage_result['total_persons']}, "
                f"Objects: {storage_result['total_objects']}"
            )
            
            # =========================================================================
            # Step 4: Update Database Status (Optional)
            # =========================================================================
            workflow.logger.info(f"📊 Step 4: Updating database status")
            
            final_result = await workflow.execute_activity(
                update_video_status_activity,
                storage_result,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=retry_policy,
            )
            
            # =========================================================================
            # Step 5: Cleanup local files (best effort)
            # =========================================================================
            try:
                await workflow.execute_activity(
                    activities_module.cleanup_files_activity,
                    {
                        "video_path": download_result.get("local_path") if download_result else None,
                        "image_path": image_result.get("local_path") if (image_result and image_result.get("status") == "downloaded") else None,
                    },
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=10),
                        backoff_coefficient=2.0,
                        maximum_attempts=2,
                    ),
                )
                workflow.logger.info("🧹 Cleanup complete for local downloaded files")
            except Exception as ce:
                workflow.logger.warning(f"⚠ Cleanup failed: {ce}")

            workflow.logger.info(f"🎉 Workflow completed successfully for video_id: {video_id}")
            
            return final_result
            
        except Exception as e:
            # Log error and return failure result
            workflow.logger.error(f"❌ Workflow failed for video_id: {video_id}: {e}")
            # Best-effort cleanup even on failures
            try:
                await workflow.execute_activity(
                    activities_module.cleanup_files_activity,
                    {
                        "video_path": download_result.get("local_path") if download_result else None,
                        "image_path": image_result.get("local_path") if (image_result and image_result.get("status") == "downloaded") else None,
                    },
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=10),
                        backoff_coefficient=2.0,
                        maximum_attempts=2,
                    ),
                )
                workflow.logger.info("🧹 Cleanup attempted after failure")
            except Exception as ce:
                workflow.logger.warning(f"⚠ Cleanup after failure also failed: {ce}")
            
            # Return error result
            return {
                "video_id": video_id,
                "status": "failed",
                "error": str(e),
                "total_persons": 0,
                "total_objects": 0,
            }


@workflow.defn
class BatchVideoProcessingWorkflow:
    """
    Workflow for processing multiple videos in batch.
    This can be useful for bulk uploads or reprocessing.
    """
    
    @workflow.run
    async def run(self, video_list: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Process multiple videos in parallel or sequence.
        
        Args:
            video_list: List of video_data dictionaries
        
        Returns:
            List of processing results
        """
        workflow.logger.info(f"🚀 Starting batch processing for {len(video_list)} videos")
        
        results = []
        
        # Process videos sequentially to avoid overwhelming the system
        # You can also use asyncio.gather() for parallel processing
        for video_data in video_list:
            try:
                # Start a child workflow for each video
                result = await workflow.execute_child_workflow(
                    VideoProcessingWorkflow.run,
                    video_data,
                    id=f"video-processing-{video_data['video_id']}",
                    task_queue="video-processing-task-queue",
                )
                results.append(result)
                
            except Exception as e:
                workflow.logger.error(f"Failed to process video {video_data['video_id']}: {e}")
                results.append({
                    "video_id": video_data['video_id'],
                    "status": "failed",
                    "error": str(e),
                })
        
        workflow.logger.info(f"🎉 Batch processing complete. Processed {len(results)} videos")
        
        return results


@workflow.defn
class VideoReprocessingWorkflow:
    """
    Workflow for reprocessing an existing video.
    Useful for updates to the ML model or fixing failed processing.
    """
    
    @workflow.run
    async def run(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reprocess a video that was already uploaded.
        
        Args:
            video_data: Video information
        
        Returns:
            Processing results
        """
        workflow.logger.info(f"🔄 Starting reprocessing workflow for video_id: {video_data['video_id']}")
        
        # Use the same processing workflow
        result = await workflow.execute_child_workflow(
            VideoProcessingWorkflow.run,
            video_data,
            id=f"video-reprocessing-{video_data['video_id']}-{workflow.now().timestamp()}",
            task_queue="video-processing-task-queue",
        )
        
        workflow.logger.info(f"✅ Reprocessing complete for video_id: {video_data['video_id']}")
        
        return result
