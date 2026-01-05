# Person Re-Identification Architecture Summary

## 🏗️ Complete Architecture Overview

### Three-Layer System

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: OBSERVATION (Qdrant)                          │
│  Immutable tracklet observations with vectors           │
├─────────────────────────────────────────────────────────┤
│  person_tracklets (Qdrant collection)                   │
│  - track_id, start_time, end_time, num_frames           │
│  - face_vec (512D), reid_vec (512D)                      │
│  - resolved_person_id (nullable, reference to PG)       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ Cross-reference
                       ↓
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: IDENTITY (PostgreSQL)                         │
│  Person hypotheses and similarity decisions             │
├─────────────────────────────────────────────────────────┤
│  persons                                                │
│  - person_id (UUID)                                     │
│  - video_id, confidence_score                           │
│  - total_tracklets, first/last_appearance_time          │
│                                                          │
│  person_tracklets (PG cache of Qdrant tracklets)        │
│  - tracklet_id, person_id, qdrant_tracklet_id           │
│  - face_quality, reid_quality (for matching)            │
│  - embedding hashes (for vector lookup)                 │
│                                                          │
│  person_tracklet_associations (Audit trail)             │
│  - person_id, tracklet_id, similarities, scores         │
│  - resolution_mode (automatic/hybrid)                   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
Video Processing Pipeline
    ↓
1. Detect & track persons
    ↓
2. Extract embeddings (face + ReID)
    ↓
3. Tracklet finishes → CREATE TRACKLET METADATA
    ↓
4. Query PostgreSQL for similar tracklets (same video)
    ↓
5. SIMILARITY SEARCH:
    ├─ Compare ReID vectors (primary)
    ├─ Compare face vectors (confirmation)
    └─ Compute fused similarity score
    ↓
6. DECISION:
    ├─ High ReID sim? → AUTO-ASSIGN to person
    ├─ Med ReID + face? → HYBRID ASSIGN to person
    └─ Low sim? → CREATE NEW PERSON
    ↓
7. INSERT INTO POSTGRESQL:
    ├─ person_tracklets (immutable record)
    ├─ person (create new or update existing)
    └─ person_tracklet_associations (audit trail)
    ↓
8. UPDATE QDRANT:
    └─ Set resolved_person_id in payload
```

---

## 📊 PostgreSQL Schema Details

### Table 1: persons
```sql
CREATE TABLE persons (
    person_id UUID PRIMARY KEY,
    video_id INTEGER,
    
    -- Identity anchors
    canonical_face_vec_id UUID,              -- Best face tracklet
    canonical_reid_vec_id UUID,              -- Best ReID tracklet
    
    -- Aggregated statistics
    confidence_score FLOAT,                  -- Avg confidence
    face_confidence_avg FLOAT,               -- Avg face quality
    reid_confidence_avg FLOAT,               -- Avg ReID quality
    total_tracklets INTEGER,                 -- Count of tracklets
    first_appearance_time TIME,              -- First tracklet start
    last_appearance_time TIME,               -- Last tracklet end
    total_appearance_duration INTERVAL,      -- Sum of all durations
    
    -- Metadata
    resolution_mode VARCHAR(30),             -- 'automatic', 'hybrid', 'pending_review'
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

**Key Points**:
- One row per unique person in a video
- `confidence_score` = average confidence across all tracklets
- `resolution_mode` indicates how identity was resolved
- Timestamps track when person was first seen and last seen

---

### Table 2: person_tracklets
```sql
CREATE TABLE person_tracklets (
    tracklet_id UUID PRIMARY KEY,
    person_id UUID,                          -- NULL until resolved
    video_id INTEGER,
    
    -- Qdrant linkage
    qdrant_tracklet_id VARCHAR(255) UNIQUE,  -- Reference to Qdrant point ID
    track_number INTEGER,                    -- Original tracker ID
    
    -- Temporal
    start_time TIME,
    end_time TIME,
    num_frames INTEGER,
    duration_sec NUMERIC(10, 2),
    
    -- Quality metrics
    face_quality NUMERIC(5, 3),              -- 0-1
    reid_quality NUMERIC(5, 3),              -- 0-1
    avg_detection_confidence NUMERIC(5, 3),
    
    -- Embedding references (not stored, just hashes)
    face_embedding_hash VARCHAR(128),        -- SHA256 hash
    reid_embedding_hash VARCHAR(128),        -- SHA256 hash
    
    -- Metadata
    attributes JSONB,                        -- hat, hood, glasses, colors
    bounding_boxes JSONB,                    -- Sample frames with boxes
    
    -- Status
    is_primary_for_person BOOLEAN,           -- Is canonical tracklet?
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

**Key Points**:
- `person_id` = NULL initially, filled after resolution
- `qdrant_tracklet_id` links back to Qdrant observation
- `track_number` = original tracker ID from video processing
- Immutable - never updated after creation
- Hashes used for fast embedding comparison (actual vectors in Qdrant)

---

### Table 3: person_tracklet_associations
```sql
CREATE TABLE person_tracklet_associations (
    association_id SERIAL PRIMARY KEY,
    person_id UUID,
    tracklet_id UUID UNIQUE,                 -- 1:1 mapping
    
    -- Matching scores
    reid_similarity NUMERIC(5, 4),           -- 0-1
    face_similarity NUMERIC(5, 4),           -- 0-1
    fused_similarity NUMERIC(5, 4),          -- Weighted combo
    
    -- Decision metadata
    resolution_mode VARCHAR(30),             -- 'automatic' or 'hybrid'
    confidence_level VARCHAR(20),            -- 'high', 'medium', 'low'
    threshold_applied NUMERIC(5, 4),         -- Which T_* was used
    
    matched_at TIMESTAMP
);
```

**Key Points**:
- Audit trail for person resolution
- One row per tracklet assignment
- Scores show why tracklet was assigned
- `confidence_level` derived from `fused_similarity`

---

### Table 4: person_appearance_timeline (Optional)
```sql
CREATE TABLE person_appearance_timeline (
    timeline_id SERIAL PRIMARY KEY,
    person_id UUID,
    video_id INTEGER,
    
    -- Continuous appearance block
    appearance_number INTEGER,               -- 1st, 2nd, 3rd appearance...
    start_time TIME,
    end_time TIME,
    duration_sec NUMERIC(10, 2),
    
    -- Gap analysis
    gap_from_previous_sec NUMERIC(10, 2),   -- Time since last appearance
    num_tracklets INTEGER,                   -- Tracklets in this block
    
    created_at TIMESTAMP
);
```

**Key Points**:
- Derived table (optional, for performance)
- Groups continuous tracklets into appearance blocks
- Useful for UI showing "person appeared 3 times, 2.5 min apart"

---

## 🎯 Decision Logic (Step-by-Step)

```python
# When new tracklet finishes:

# 1. Check embedding quality
if tracklet.reid_quality < 0.6:
    → CREATE NEW PERSON (poor quality)
    
# 2. Search for candidates in same video
candidates = query_existing_tracklets(video_id)

# 3. For each candidate, compute similarities
for candidate in candidates:
    reid_sim = cosine_similarity(tracklet.reid_vec, candidate.reid_vec)
    face_sim = cosine_similarity(tracklet.face_vec, candidate.face_vec)
    fused_sim = 0.65 * reid_sim + 0.35 * face_sim

# 4. Apply decision thresholds
best_match = candidates[max_fused_sim]

if best_match.reid_sim >= 0.75:
    → AUTO-ASSIGN (automatic mode)
    → confidence_level = "high"
    
elif best_match.reid_sim >= 0.60 AND best_match.face_sim >= 0.55:
    → HYBRID-ASSIGN (face confirms ReID)
    → confidence_level = "medium"
    
else:
    → CREATE NEW PERSON
    → confidence_level = "low"

# 5. Store decision in PostgreSQL
insert_tracklet(tracklet_id, person_id, similarities, mode)
```

---

## 📈 Thresholds & Configuration

```python
# Identity Resolution Thresholds
REID_WEIGHT = 0.65      # ReID weighted 65% (more reliable)
FACE_WEIGHT = 0.35      # Face weighted 35% (confirmation)

# High confidence (automatic assignment)
T_HIGH_REID = 0.75
T_HIGH_FACE = 0.70
FUSED_HIGH = 0.72

# Medium confidence (hybrid - ReID + face confirmation)
T_MED_REID = 0.60
T_MED_FACE = 0.55
FUSED_MED = 0.62

# Low confidence (new person if below this)
T_LOW_REID = 0.50
T_LOW_FACE = 0.45
FUSED_LOW = 0.50

# Quality requirements
MIN_FACE_QUALITY = 0.5  # Must be detectable
MIN_REID_QUALITY = 0.6  # Must be recognizable
```

---

## 🔄 Comparison: Objects vs Persons

| Aspect | Objects (Qdrant mutation) | Persons (PostgreSQL identity) |
|--------|---------------------------|-------------------------------|
| **Storage Strategy** | One Qdrant point, payload updated | Immutable Qdrant + PostgreSQL identity |
| **Mutation** | ✅ Merge tracklets into one payload | ❌ Never mutate, only append |
| **Reappearance** | Array in payload: `reappearances[]` | Separate rows in `person_tracklets` |
| **Identity** | Weak (can be lost if point deleted) | Strong (persisted in PostgreSQL) |
| **Auditability** | Limited (overwrites history) | Complete (all associations logged) |
| **Scalability** | Low (payload grows unbounded) | High (distributed across tables) |
| **FYP Defensibility** | Pragmatic but not ideal | Architecturally sound, industry-standard |

---

## 🚀 Why This Architecture Wins

1. **Separation of Concerns**
   - Qdrant = observations (immutable, vectorized)
   - PostgreSQL = identity (mutable hypothesis, relational)

2. **Auditability**
   - Every person assignment tracked in `person_tracklet_associations`
   - Can trace exact similarity scores and thresholds used

3. **Extensibility**
   - Easy to add new matching algorithms
   - Easy to add manual review (just change threshold)
   - Easy to implement cross-video matching

4. **Performance**
   - Tracklets cached in PostgreSQL (fast searches)
   - Vectors remain in Qdrant (efficient similarity search)
   - Best of both worlds

5. **Viva Defensibility**
   - Clear architectural reasoning
   - Industry-standard pattern (observations vs identity)
   - Shows maturity and foresight
   - Can explain all trade-offs

---

## 📋 SQL Queries for Analytics

### All appearances of a person
```sql
SELECT start_time, end_time, duration_sec, num_frames
FROM person_tracklets
WHERE person_id = $1
ORDER BY start_time;
```

### Appearance timeline with gaps
```sql
SELECT 
    tracklet_id, start_time, end_time,
    EXTRACT(EPOCH FROM (start_time - LAG(end_time) OVER (ORDER BY start_time))) as gap_sec
FROM person_tracklets
WHERE person_id = $1
ORDER BY start_time;
```

### Video person statistics
```sql
SELECT
    COUNT(DISTINCT person_id) as total_persons,
    AVG(tracklet_count) as avg_tracklets_per_person,
    MAX(tracklet_count) as max_tracklets
FROM (
    SELECT person_id, COUNT(*) as tracklet_count
    FROM person_tracklets
    WHERE video_id = $1
    GROUP BY person_id
);
```

### Persons with reappearances
```sql
SELECT 
    person_id, 
    COUNT(*) as tracklet_count,
    MAX(end_time) - MIN(start_time) as total_span
FROM person_tracklets
WHERE video_id = $1
GROUP BY person_id
HAVING COUNT(*) > 1
ORDER BY tracklet_count DESC;
```

---

## 🎓 Viva Talking Points

1. **Why separate Qdrant and PostgreSQL?**
   - "Qdrant handles vectorized observations, PostgreSQL manages the identity hypothesis. This separation allows us to evolve matching algorithms independently."

2. **Why immutable tracklets?**
   - "Immutability provides auditability. Every assignment is logged with similarity scores, enabling forensic analysis and continuous improvement."

3. **Why ReID primary + face confirmation?**
   - "ReID embeddings are more robust to pose/angle variations. Face is higher resolution but sensitive to lighting. Using both provides reliability."

4. **How do you handle false matches?**
   - "Thresholds are tuned conservatively. Borderline matches create new persons rather than false merges. We can always merge later with more data."

5. **How does this scale?**
   - "PostgreSQL indexes on video_id and person_id enable fast searches. Qdrant vectors stay off-heap. The system is horizontally scalable."

---

## ✅ Implementation Checklist

- [ ] Create PostgreSQL migration file (004_person_identity_tables.sql)
- [ ] Run migration on backend database
- [ ] Import `person_reid_manager.py` into pipeline
- [ ] Integrate into `insert_tracklet_to_qdrant()` workflow
- [ ] Add PostgreSQL connection to pipeline environment
- [ ] Update `.env` with PostgreSQL credentials
- [ ] Test person matching with sample videos
- [ ] Create analytics dashboard queries
- [ ] Document thresholds in README
- [ ] Prepare viva presentation

---

## 📚 Files Created

1. **SQL Migration**: `migrations/004_person_identity_tables.sql`
2. **Python Module**: `pipeline/person_reid_manager.py`
3. **Integration Guide**: `PERSON_REID_INTEGRATION.md`
4. **This Summary**: `PERSON_REID_ARCHITECTURE.md`
