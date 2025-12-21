import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, models

# --- 1. Environment and Constants ---

# Load environment variables from .env file (for QDRANT_HOST, QDRANT_PORT)
load_dotenv() 

# Define Collection Names
PERSON_TRACKS_COLLECTION = "person_tracks"
OBJECT_TRACKS_COLLECTION = "object_tracks"

# Define Vector Dimensions
DIM_FACE = 512           # InsightFace ArcFace embedding
DIM_REID = 512           # TorchReID OSNet embedding
DIM_MULTI = 768          # CLIP for text search (multi-modal)
DIM_OBJECT = 768         # CLIP for object crops

# Define Vector Parameter Objects (Using Cosine distance is standard for similarity)
VECTOR_PARAMS_512 = VectorParams(size=DIM_FACE, distance=Distance.COSINE)
VECTOR_PARAMS_768 = VectorParams(size=DIM_MULTI, distance=Distance.COSINE)

# --- 2. Named Vector Definitions ---

# Person Tracks: face_vec (512D), reid_vec (512D), multi_vec (768D for CLIP)
PERSON_TRACKS_VECTORS = {
    "face_vec": VECTOR_PARAMS_512,  # Face embedding from InsightFace (512D)
    "reid_vec": VECTOR_PARAMS_512,  # Body ReID embedding from TorchReID (512D)
    "multi_vec": VECTOR_PARAMS_768  # CLIP embedding for text search (768D)
}

# Object Tracks: object_vec (768D for CLIP), multi_vec (768D shared with person_tracks)
OBJECT_TRACKS_VECTORS = {
    "object_vec": VECTOR_PARAMS_768,  # CLIP embedding of object crop (768D)
    "multi_vec": VECTOR_PARAMS_768    # CLIP for text search, shared dimension (768D)
}


def create_qdrant_schema(client: QdrantClient):
    """
    Creates person_tracks and object_tracks collections with their
    respective named vectors and critical payload indexes.
    
    PERSON_TRACKS: Stores one record per track (all face/ReID embeddings for that track)
    OBJECT_TRACKS: Stores one record per object detection (linked to person_tracks via associated_person_track)
    """
    print("Starting Qdrant collection setup and indexing...")
    
    # --- 3. Collection Creation (PRESERVE EXISTING DATA) ---

    # A. Create person_tracks collection if it doesn't exist
    try:
        existing_collections = client.get_collections()
        collection_names = [col.name for col in existing_collections.collections]

        if PERSON_TRACKS_COLLECTION not in collection_names:
            client.create_collection(
                collection_name=PERSON_TRACKS_COLLECTION,
                vectors_config=PERSON_TRACKS_VECTORS
            )
            print(f"✅ Collection '{PERSON_TRACKS_COLLECTION}' created with vectors: face_vec (512D), reid_vec (512D), multi_vec (768D)")
        else:
            print(f"ℹ️ Collection '{PERSON_TRACKS_COLLECTION}' already exists - preserving existing data")
    except Exception as e:
        print(f"⚠ Error creating/checking person_tracks collection: {e}")
        return False

    # B. Create object_tracks collection if it doesn't exist
    try:
        existing_collections = client.get_collections()
        collection_names = [col.name for col in existing_collections.collections]

        if OBJECT_TRACKS_COLLECTION not in collection_names:
            client.create_collection(
                collection_name=OBJECT_TRACKS_COLLECTION,
                vectors_config=OBJECT_TRACKS_VECTORS
            )
            print(f"✅ Collection '{OBJECT_TRACKS_COLLECTION}' created with vectors: object_vec (768D), multi_vec (768D)")
        else:
            print(f"ℹ️ Collection '{OBJECT_TRACKS_COLLECTION}' already exists - preserving existing data")
    except Exception as e:
        print(f"⚠ Error creating/checking object_tracks collection: {e}")
        return False

    # --- 4. Payload Indexing (CRITICAL for Hybrid Search Performance) ---

    try:
        # Indexes for person_tracks
        print(f"\nSetting up person_tracks indexes...")
        
        # INTEGER indexes
        person_index_fields = [
            ("video_id", models.PayloadSchemaType.INTEGER),
            ("segment_id", models.PayloadSchemaType.INTEGER),
            ("track_id", models.PayloadSchemaType.INTEGER),
            ("person_gender", models.PayloadSchemaType.KEYWORD),
            ("upper_color", models.PayloadSchemaType.KEYWORD),
            ("lower_color", models.PayloadSchemaType.KEYWORD),
            ("object_carried", models.PayloadSchemaType.KEYWORD),
            ("avg_confidence", models.PayloadSchemaType.FLOAT)
        ]
        
        for field, index_type in person_index_fields:
            try:
                client.create_payload_index(PERSON_TRACKS_COLLECTION, field, index_type)
            except Exception as e:
                # Index likely already exists - continue silently
                pass
        
        print(f"✓ Person tracks indexes ready")
    except Exception as e:
        print(f"⚠ Error setting up person_tracks indexes: {e}")

    try:
        # Indexes for object_tracks
        print(f"Setting up object_tracks indexes...")
        
        object_index_fields = [
            ("video_id", models.PayloadSchemaType.INTEGER),
            ("segment_id", models.PayloadSchemaType.INTEGER),
            ("object_id", models.PayloadSchemaType.INTEGER),
            ("associated_person_track", models.PayloadSchemaType.INTEGER),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("object_color", models.PayloadSchemaType.KEYWORD)
        ]
        
        for field, index_type in object_index_fields:
            try:
                client.create_payload_index(OBJECT_TRACKS_COLLECTION, field, index_type)
            except Exception as e:
                # Index likely already exists - continue silently
                pass
        
        print(f"✓ Object tracks indexes ready")
    except Exception as e:
        print(f"⚠ Error setting up object_tracks indexes: {e}")
    
    print("\n" + "="*60)
    print("✅ Qdrant schema setup and indexing complete!")
    print("="*60)


# if __name__ == "__main__":
#     # --- Client Initialization ---
#     QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
#     # Using gRPC port 6334 for client communication, which is faster.
#     QDRANT_PORT = int(os.environ.get("QDRANT_GRPC_PORT", 6334)) 
    
#     try:
#         # Connect to Qdrant using configuration
#         client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
#         print(f"Attempting to connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        
#         # Simple health check before creating collections
#         client.get_collections()
#         print("Connection successful.")
        
#         create_qdrant_schema(client)

#     except Exception as e:
#         print(f"\n--- ERROR ---")
#         print(f"Could not connect to Qdrant or create collections.")
#         print(f"Please ensure your Docker stack is running and QDRANT_GRPC_PORT is correct.")
#         print(f"Details: {e}", file=sys.stderr)
#         sys.exit(1)