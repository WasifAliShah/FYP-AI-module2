# 🚀 Quick Start Guide - Video Processing with Temporal

## Prerequisites

- Python 3.8+ installed
- Node.js 18+ installed
- Docker installed
- PostgreSQL running (for VisionIndex backend)
- MongoDB running (for VisionIndex backend)

## Step 1: Start Infrastructure Services

### Option A: Using Docker Compose (Recommended)

```bash
cd "FYP AI 2/video_processing_temporal"
docker-compose up -d
```

This starts:
- ✅ Temporal Server (port 7233)
- ✅ Temporal Web UI (port 8233)
- ✅ Qdrant Vector DB (port 6333, 6334)

Verify services:
```bash
docker ps
```

### Option B: Start Services Individually

**Temporal:**
```bash
docker run -d -p 7233:7233 -p 8233:8233 --name temporal temporalio/auto-setup:latest
```

**Qdrant:**
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest
```

## Step 2: Set Up Python Environment

```bash
cd "FYP AI 2/video_processing_temporal"

# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configure Environment Variables

Copy and configure the environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
```env
S3_ENDPOINT=https://s3.us-east-005.backblazeb2.com
S3_REGION=us-east-005
S3_ACCESS_KEY_ID=your_actual_key
S3_SECRET_ACCESS_KEY=your_actual_secret
S3_BUCKET_NAME=visionindex-video-s3

QDRANT_URL=http://localhost:6333
```

## Step 4: Set Up Backend

```bash
cd "../../VisionIndex-Backend/app"

# Install dependencies (includes @temporalio/client)
npm install

# Verify .env has Temporal configuration
# Should have: TEMPORAL_HOST=localhost:7233
```

## Step 5: Start the Worker

Open a new terminal:

```bash
cd "FYP AI 2/video_processing_temporal"

# Activate venv if you created one
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Start the worker
python worker.py
```

You should see:
```
✅ Connected to Temporal server
🎬 Video Processing Worker Started!
================================================================================
Waiting for workflow tasks...
```

✅ **Keep this terminal open** - the worker needs to run continuously!

## Step 6: Start the Backend

Open another terminal:

```bash
cd "VisionIndex-Backend/app"

# Start backend
npm run dev
```

You should see:
```
✅ Temporal client initialized
Server is running on port 3000
```

## Step 7: Test the System

### Option A: Upload Video via Frontend

1. Start the frontend:
   ```bash
   cd VisionIndex-Frontend
   npm run dev
   ```

2. Open http://localhost:5173
3. Login and upload a video
4. Watch the worker terminal for progress!

### Option B: Test with API

```bash
# Get upload URL
curl -X POST http://localhost:3000/api/videos/upload-url \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "fileName": "test.mp4",
    "fileSize": 1048576,
    "fileType": "video/mp4"
  }'

# After uploading to S3, register the video
curl -X POST http://localhost:3000/api/videos/register \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "fileName": "videos/123/test.mp4",
    "originalName": "test.mp4",
    "fileSize": 1048576,
    "checksum": "abc123"
  }'
```

### Option C: Test with Python Script

Edit `trigger_workflow.py` with your test video data:

```python
video_data = {
    "video_id": 999,
    "file_name": "videos/94/test_video.mp4",  # Your S3 key
    "original_name": "test_video.mp4",
    "uploader_id": 1,
    "storage_path": "https://s3.example.com/videos/94/test_video.mp4",
}
```

Run it:
```bash
python trigger_workflow.py
```

## Step 8: Monitor Progress

### Temporal Web UI
Open http://localhost:8233

You'll see:
- All running workflows
- Workflow execution history
- Activity logs and details
- Any errors or retries

### Check Video Status
```bash
curl http://localhost:3000/api/videos/123/workflow-status \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Check Worker Logs
Watch the worker terminal for real-time progress:
```
📥 Step 1: Downloading video...
✅ Download complete
🎬 Step 2: Processing video...
✅ Processing complete
💾 Step 3: Verifying Qdrant storage...
✅ Results verified - Persons: 15, Objects: 8
```

## Troubleshooting

### "Failed to connect to Temporal server"
```bash
# Check if Temporal is running
docker ps | grep temporal

# Restart if needed
docker restart temporal
```

### "Failed to download video"
- Check S3 credentials in `.env`
- Verify the video exists in your bucket
- Check network connectivity

### "Module not found" errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Backend can't connect to Temporal
```bash
# Verify backend .env has:
TEMPORAL_HOST=localhost:7233

# Restart backend
npm run dev
```

## What Happens During Video Processing?

1. 📤 **User uploads video** → Stored in S3
2. 📝 **Backend registers** → Creates DB record
3. 🚀 **Workflow starts** → Temporal orchestrates
4. 📥 **Download** → Worker downloads from S3
5. 🎬 **Process** → ML pipeline analyzes video
   - Face detection (YOLO)
   - Face recognition (InsightFace)
   - Person tracking (DeepSORT)
   - Object detection
   - Attribute extraction
6. 💾 **Store** → Results saved to Qdrant
   - Person embeddings (face, ReID, CLIP)
   - Object embeddings
   - Metadata (colors, attributes, objects)
7. ✅ **Complete** → Video status updated
8. 🔍 **Search** → Users can search by text/attributes

## Monitoring Best Practices

### Worker Health
- Keep the worker terminal open and visible
- Watch for "heartbeat" messages during long operations
- Check for error messages

### Temporal Dashboard
- Monitor workflow success rate
- Check average execution time
- Review failed workflows

### Database
- Check PostgreSQL for video status
- Check MongoDB for processing metadata
- Check Qdrant for stored embeddings

## Next Steps

1. ✅ System running → Upload test videos
2. 📊 Monitor workflows → Watch Temporal UI
3. 🔍 Test search → Query processed videos
4. 📈 Scale up → Add more workers if needed

## Production Checklist

Before deploying to production:

- [ ] Use Temporal Cloud (not Docker)
- [ ] Set up proper authentication
- [ ] Configure TLS for all connections
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backup for Qdrant
- [ ] Set up log aggregation
- [ ] Test failure scenarios
- [ ] Document recovery procedures

## Support

If you encounter issues:

1. Check this guide first
2. Review the detailed README.md
3. Check Temporal Web UI for errors
4. Check worker logs
5. Check backend logs

## Common Commands

```bash
# Start everything
docker-compose up -d
python worker.py  # In one terminal
npm run dev       # In another terminal

# Stop everything
docker-compose down
Ctrl+C in worker terminal
Ctrl+C in backend terminal

# Restart worker
Ctrl+C
python worker.py

# View logs
docker logs temporal
docker logs qdrant

# Check service health
curl http://localhost:7233
curl http://localhost:6333
curl http://localhost:3000/health
```

---

🎉 **Congratulations!** Your video processing system with Temporal is now running!
