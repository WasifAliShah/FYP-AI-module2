import os
import requests
import boto3
from botocore.client import Config
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

ACCESS_KEY = os.getenv("S3_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
BUCKET = os.getenv("S3_BUCKET_NAME")
ENDPOINT = os.getenv("S3_ENDPOINT")
REGION = os.getenv("S3_REGION")

# Create S3 client
s3 = boto3.client(
    "s3",
    region_name=REGION,
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4"),
)


def get_signed_download_url(file_name, expires=3600):
    """Generate a presigned download URL for an S3 file."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": file_name},
        ExpiresIn=expires,
    )


def download_video(file_name):
    """Download the video using a presigned URL and save it to downloaded_videos/."""
    
    # Generate signed URL
    url = get_signed_download_url(file_name)

    # Ensure output folder exists
    output_folder = "downloaded_videos"
    os.makedirs(output_folder, exist_ok=True)

    # Output path
    output_path = os.path.join(output_folder, os.path.basename(file_name))

    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")

    # Stream download
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("✅ Download complete:", output_path)
    return output_path


# ---------------------------
# Example usage:
# ---------------------------

file_name = "videos/94/1764423168535_qmxes7itg2e.mp4"  # <-- replace with your file key
download_video(file_name)
