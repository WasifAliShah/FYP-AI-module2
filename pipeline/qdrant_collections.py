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
DIM_MULTI = 512          # CLIP ViT-B/32 for text search (multi-modal)
DIM_OBJECT = 512         # CLIP ViT-B/32 for object crops

# Define Vector Parameter Objects (Using Cosine distance is standard for similarity)
VECTOR_PARAMS_512 = VectorParams(size=DIM_FACE, distance=Distance.COSINE)

# --- 2. Named Vector Definitions ---

# Person Tracks: face_vec (512D), reid_vec (512D), multi_vec (512D for CLIP ViT-B/32)
PERSON_TRACKS_VECTORS = {
    "face_vec": VECTOR_PARAMS_512,  # Face embedding from InsightFace (512D)
    "reid_vec": VECTOR_PARAMS_512,  # Body ReID embedding from TorchReID (512D)
    "multi_vec": VECTOR_PARAMS_512  # CLIP ViT-B/32 embedding for text search (512D)
}

# Object Tracks: object_vec (512D for CLIP ViT-B/32), multi_vec (512D shared with person_tracks)
OBJECT_TRACKS_VECTORS = {
    "object_vec": VECTOR_PARAMS_512,  # CLIP ViT-B/32 embedding of object crop (512D)
    "multi_vec": VECTOR_PARAMS_512    # CLIP ViT-B/32 for text search, shared dimension (512D)
}


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection already exists."""
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def create_qdrant_schema(client: QdrantClient, realtime_mode: bool = True):
    """
    Creates or validates person_tracks and object_tracks collections.
    
    IMPORTANT: This function NOW PRESERVES existing data!
    - If collection exists: Keep all previous data, just create indexes if needed
    - If collection doesn't exist: Create fresh collection
    
    Args:
        client: QdrantClient instance
        realtime_mode: If True, creates collections and indexes (REAL-TIME mode).
                      If False, only validates collections exist (POST-PROCESSING mode).
    
    PERSON_TRACKS: Stores one record per track (all face/ReID embeddings for that track)
    OBJECT_TRACKS: Stores one record per object detection (linked to person_tracks via associated_person_track)
    """
    
    if not realtime_mode:
        # POST-PROCESSING MODE: Only check if collections exist
        print("POST-PROCESSING MODE: Validating Qdrant collections...")
        person_exists = collection_exists(client, PERSON_TRACKS_COLLECTION)
        object_exists = collection_exists(client, OBJECT_TRACKS_COLLECTION)
        
        if person_exists:
            print(f"✅ Collection '{PERSON_TRACKS_COLLECTION}' exists")
        else:
            print(f"⚠ ERROR: Collection '{PERSON_TRACKS_COLLECTION}' not found! Create it in REAL-TIME mode first.")
            return False
        
        if object_exists:
            print(f"✅ Collection '{OBJECT_TRACKS_COLLECTION}' exists")
        else:
            print(f"⚠ ERROR: Collection '{OBJECT_TRACKS_COLLECTION}' not found! Create it in REAL-TIME mode first.")
            return False
        
        print("✅ All collections validated successfully!")
        return True
    
    # REAL-TIME MODE: Create collections (if not exist) and indexes - PRESERVE EXISTING DATA
    print("REAL-TIME MODE: Starting Qdrant collection setup and indexing (PRESERVING existing data)...")
    
    # --- 3. Collection Creation (CREATE ONLY IF NOT EXISTS) ---

    # A. Create person_tracks (if it doesn't exist)
    try:
        if collection_exists(client, PERSON_TRACKS_COLLECTION):
            print(f"✅ Collection '{PERSON_TRACKS_COLLECTION}' already exists - PRESERVING {client.count(PERSON_TRACKS_COLLECTION).count} records")
        else:
            client.create_collection(
                collection_name=PERSON_TRACKS_COLLECTION,
                vectors_config=PERSON_TRACKS_VECTORS
            )
            print(f"✅ Collection '{PERSON_TRACKS_COLLECTION}' created with vectors: face_vec (512D), reid_vec (512D), multi_vec (512D)")
    except Exception as e:
        print(f"⚠ Error with person_tracks collection: {e}")
        return False

    # B. Create object_tracks (if it doesn't exist)
    try:
        if collection_exists(client, OBJECT_TRACKS_COLLECTION):
            print(f"✅ Collection '{OBJECT_TRACKS_COLLECTION}' already exists - PRESERVING {client.count(OBJECT_TRACKS_COLLECTION).count} records")
        else:
            client.create_collection(
                collection_name=OBJECT_TRACKS_COLLECTION,
                vectors_config=OBJECT_TRACKS_VECTORS
            )
            print(f"✅ Collection '{OBJECT_TRACKS_COLLECTION}' created with vectors: object_vec (512D), multi_vec (512D)")
    except Exception as e:
        print(f"⚠ Error with object_tracks collection: {e}")
        return False

    # --- 4. Payload Indexing (CRITICAL for Hybrid Search Performance) ---

    try:
        # Indexes for person_tracks
        print(f"\nIndexing person_tracks payload fields...")
        
        # INTEGER indexes
        try:
            client.create_payload_index(PERSON_TRACKS_COLLECTION, "video_id", models.PayloadSchemaType.INTEGER)
        except:
            pass  # Index may already exist
        try:
            client.create_payload_index(PERSON_TRACKS_COLLECTION, "segment_id", models.PayloadSchemaType.INTEGER)
        except:
            pass
        try:
            client.create_payload_index(PERSON_TRACKS_COLLECTION, "track_id", models.PayloadSchemaType.INTEGER)
        except:
            pass
        
        # KEYWORD indexes (for filtering)
        try:
            client.create_payload_index(PERSON_TRACKS_COLLECTION, "person_gender", models.PayloadSchemaType.KEYWORD)
        except:
            pass
        try:
            client.create_payload_index(PERSON_TRACKS_COLLECTION, "upper_color", models.PayloadSchemaType.KEYWORD)
        except:
            pass
        try:
            client.create_payload_index(PERSON_TRACKS_COLLECTION, "lower_color", models.PayloadSchemaType.KEYWORD)
        except:
            pass
        try:
            client.create_payload_index(PERSON_TRACKS_COLLECTION, "object_carried", models.PayloadSchemaType.KEYWORD)
        except:
            pass
        
        # FLOAT index
        try:
            client.create_payload_index(PERSON_TRACKS_COLLECTION, "avg_confidence", models.PayloadSchemaType.FLOAT)
        except:
            pass
        
        print(f"✅ Indexes ready for {PERSON_TRACKS_COLLECTION}")
    except Exception as e:
        print(f"⚠ Error with person_tracks indexes: {e}")

    try:
        # Indexes for object_tracks
        print(f"\nIndexing object_tracks payload fields...")
        
        # INTEGER indexes
        try:
            client.create_payload_index(OBJECT_TRACKS_COLLECTION, "video_id", models.PayloadSchemaType.INTEGER)
        except:
            pass
        try:
            client.create_payload_index(OBJECT_TRACKS_COLLECTION, "segment_id", models.PayloadSchemaType.INTEGER)
        except:
            pass
        try:
            client.create_payload_index(OBJECT_TRACKS_COLLECTION, "object_id", models.PayloadSchemaType.INTEGER)
        except:
            pass
        try:
            client.create_payload_index(OBJECT_TRACKS_COLLECTION, "associated_person_track", models.PayloadSchemaType.INTEGER)
        except:
            pass
        
        # KEYWORD indexes
        try:
            client.create_payload_index(OBJECT_TRACKS_COLLECTION, "category", models.PayloadSchemaType.KEYWORD)
        except:
            pass
        try:
            client.create_payload_index(OBJECT_TRACKS_COLLECTION, "object_color", models.PayloadSchemaType.KEYWORD)
        except:
            pass
        
        print(f"✅ Indexes ready for {OBJECT_TRACKS_COLLECTION}")
    except Exception as e:
        print(f"⚠ Error with object_tracks indexes: {e}")
    
    return True


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