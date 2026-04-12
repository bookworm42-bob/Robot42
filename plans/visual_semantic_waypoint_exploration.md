# Visual Semantic Waypoint Exploration Plan

## Goal

Build a semantic-place layer that remembers useful named apartment waypoints such as `kitchen`, `dining_area`, `living_room`, `desk_area`, `bedroom`, `bathroom_entry`, and similar places.

This layer must be driven by RGB-D visual evidence and map projection, not by random or uniform sampling of free space.

The end result should be:

- a reliable 2D occupancy map for navigation
- frontier memory for continued exploration
- named semantic place anchors for later navigation requests
- evidence explaining why each named place was created

Example final named place:

```json
{
  "place_id": "place_kitchen_001",
  "label": "kitchen",
  "anchor_pose": {"x": -1.55, "y": 1.20, "yaw": 2.85},
  "evidence_pose": {"x": -2.25, "y": 1.35, "yaw": 0.0},
  "confidence": 0.82,
  "source_evidence_ids": ["sem_ev_012", "sem_ev_019"],
  "source_frame_ids": ["kf_004", "kf_011"],
  "evidence": [
    "cabinet-like objects visible",
    "counter-like horizontal surface visible",
    "evidence projects to left-side mapped area"
  ],
  "status": "provisional"
}
```

## Non-Goals

Do not make semantic waypoints a second exploration algorithm.

Do not use random points in free space as semantic waypoints.

Do not generate semantic waypoint candidates from geometry alone.

Do not make semantic waypoints compete with frontiers during normal exploration.

Do not let semantic code change frontier generation, frontier memory, frontier reachability, Nav2 behavior, or occupancy-map fusion.

Do not ask the LLM to invent map coordinates from vibes. The LLM may interpret labels and evidence, but coordinates must be grounded through deterministic RGB-D projection and reachability checks.

Do not navigate to semantic anchors during map discovery unless explicitly requested by an operator or a separate semantic inspection mode.

## Critical Separation From Frontier Exploration

Frontier exploration answers:

> Where should the robot go next to discover more navigable map?

Visual semantic waypointing answers:

> What useful named places has the robot visually observed, and where is a safe reachable pose to stand near them later?

These are separate responsibilities.

Frontier code owns:

- RGB-D occupancy integration
- 2D map update
- frontier detection
- frontier memory
- frontier deduplication
- frontier reachability
- frontier completion/failure
- Nav2 navigation to selected frontier targets

Semantic code owns:

- visual semantic evidence extraction from RGB/RGB-D frames
- projecting semantic evidence into map coordinates
- snapping safe anchor poses near semantic evidence
- clustering duplicate evidence
- updating named place memory
- final semantic consolidation after exploration

Semantic code may read the occupancy map and keyframes. It must not mutate frontier memory or run frontier generation.

## Correct High-Level Pipeline

The semantic layer should be passive during exploration.

```text
frontier exploration produces scan/keyframe
  -> semantic evidence extractor analyzes RGB/RGB-D frame
  -> detected semantic cues are projected into map coordinates
  -> deterministic code finds a safe reachable anchor near the evidence
  -> duplicate evidence is merged into semantic memory
  -> LLM/VLM validates or updates named place memory
```

After frontier exploration is complete or mostly complete:

```text
final map + all evidence clusters + source images
  -> LLM/VLM consolidation pass
  -> final named places and navigation anchors
```

This means semantic processing should observe the exploration stream. It should not decide where the robot goes next while the robot is still discovering navigable space.

The only normal high-level exploration target should remain a frontier selected from frontier information.

Semantic named places become useful after or alongside exploration as remembered destinations:

- "go to kitchen"
- "go to dining area"
- "go to desk area"
- "go to living room"
- "go to bathroom entrance"

## Data Model

### Semantic Evidence

Semantic evidence is where the meaningful visual thing is. It is not necessarily navigable.

```python
@dataclass
class SemanticEvidence:
    evidence_id: str
    label_hint: str
    evidence_pose: Pose2D
    source_frame_ids: list[str]
    source_pixels: list[PixelRegion]
    confidence: float
    evidence: list[str]
    status: str  # provisional, accepted, rejected, merged
```

Examples of `label_hint`:

- `kitchen`
- `dining_area`
- `living_room`
- `desk_area`
- `bedroom`
- `bathroom_entry`
- `storage_area`
- `hallway`
- `unknown_semantic_area`

### Pixel Region

Pixel regions ground visual detections back to the image.

```python
@dataclass
class PixelRegion:
    frame_id: str
    bbox_xyxy: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    depth_m: float | None
    image_position: str  # left, center, right, upper_left, lower_right, etc.
    object_label: str | None
    description: str
```

The system can start with coarse `center_uv` and `image_position`. Later it can use segmentation masks.

### Semantic Anchor Candidate

Anchor candidates are safe robot poses near evidence, not on top of objects.

```python
@dataclass
class SemanticAnchorCandidate:
    anchor_id: str
    label_hint: str
    anchor_pose: Pose2D
    evidence_pose: Pose2D
    source_evidence_ids: list[str]
    source_frame_ids: list[str]
    confidence: float
    reachability_status: str  # reachable, unreachable, unknown
    free_space_path_distance_m: float | None
    line_of_sight_score: float
    evidence: list[str]
    status: str  # provisional, accepted, rejected, merged
```

### Named Place

Named places are final or provisional navigation memory entries.

```python
@dataclass
class NamedPlace:
    place_id: str
    label: str
    anchor_pose: Pose2D
    evidence_pose: Pose2D | None
    source_anchor_ids: list[str]
    source_evidence_ids: list[str]
    source_frame_ids: list[str]
    confidence: float
    evidence: list[str]
    notes: list[str]
    status: str  # provisional, confirmed, rejected
```

The robot navigates to `anchor_pose`, not `evidence_pose`.

## RGB-D Pixel To Map Grounding

The LLM/VLM can identify semantically meaningful pixels, but deterministic code must convert those pixels into map coordinates.

For each RGB-D frame:

1. Get camera pose in map frame.
2. Get camera intrinsics.
3. For each selected pixel or mask:
   - read depth at the pixel
   - if depth is invalid, use median depth around the pixel or ask the model for near/mid/far as fallback
   - back-project pixel to camera-frame 3D point
   - transform camera-frame point into map/world frame
   - project world point to 2D map coordinate
4. Store this as `evidence_pose`.

For a coarse first version, the VLM may output:

```json
{
  "label_hint": "kitchen",
  "visual_cues": ["counter", "cabinets"],
  "image_region": "left",
  "depth_hint": "mid",
  "confidence": 0.68
}
```

Then deterministic code can:

- choose representative pixels in the left/mid image region
- use median valid depth from that region
- project a rough evidence point into the map

This is less accurate than object masks, but still grounded by RGB-D geometry.

The correct direction of authority is:

```text
LLM/VLM says: "this image region appears to contain a kitchen counter"
deterministic code says: "that image region projects to this map coordinate"
deterministic code says: "this reachable free-space pose can observe that coordinate"
LLM/VLM says: "this evidence should create/update/merge/reject a named place"
```

The incorrect direction is:

```text
LLM/VLM says: "put kitchen waypoint at x=-1.2, y=3.4"
```

Direct LLM-generated coordinates must be rejected or treated only as low-confidence hints for debugging.

## Evidence Pose Versus Anchor Pose

The evidence pose is where the object/semantic cue is.

The anchor pose is where the robot should stand.

The robot must not navigate to the evidence point if that point is on a couch, table, cabinet, counter, wall, bed, TV, or shelf.

Anchor selection should:

1. Start from `evidence_pose`.
2. Search nearby known free cells within a radius, for example `0.6m` to `2.5m`.
3. Reject occupied cells, unknown cells, and footprint-colliding cells.
4. Reject cells unreachable through known free space.
5. Prefer cells with line of sight to the evidence point.
6. Prefer cells at a comfortable observation distance.
7. Set yaw to face the evidence point.
8. If no valid anchor exists, keep the evidence but mark the anchor candidate `unreachable`.

Scoring example:

```text
score =
  + visual_confidence
  + line_of_sight_score * 0.4
  + free_space_clearance_score * 0.3
  - abs(distance_to_evidence_m - ideal_distance_m) * 0.2
  - path_distance_from_robot_m * 0.05
```

Ideal observation distance can be around `1.0m` to `1.8m` for indoor objects.

## Multiple Objects In One Region

The LLM/VLM should identify multiple semantic cues in the same area.

Example kitchen evidence:

- cabinet
- counter
- sink
- stove
- fridge

Example dining evidence:

- table
- chair
- dining chair cluster

Example living room evidence:

- couch
- TV
- coffee table
- rug

The system should project each cue into map coordinates, then cluster them.

Cluster rule:

- same or compatible label hints
- evidence poses within a spatial radius, for example `1.5m` to `3.0m`
- overlapping source frames or adjacent camera poses
- semantically compatible object labels

For a cluster, create one anchor pose that can observe the common region.

Anchor for a cluster should not be at the average object coordinate. It should be a reachable free-space observation point near the cluster, selected by line-of-sight and clearance.

Example:

```json
{
  "label_hint": "kitchen",
  "cluster_evidence": [
    {"object_label": "counter", "evidence_pose": {"x": -2.3, "y": 1.1}},
    {"object_label": "cabinet", "evidence_pose": {"x": -2.4, "y": 1.5}},
    {"object_label": "sink", "evidence_pose": {"x": -2.1, "y": 1.3}}
  ],
  "anchor_pose": {"x": -1.5, "y": 1.25, "yaw": 3.05}
}
```

This matters because an apartment region is usually defined by several objects, not one pixel.

Examples:

- A `kitchen` may be the common region around counter, sink, cabinets, oven, and fridge.
- A `dining_area` may be the common region around table, chairs, pendant light, rug, or island seating.
- A `living_room` may be the common region around couch, TV, coffee table, media console, and rug.
- A `desk_area` may be the common region around desk, monitor, office chair, keyboard, shelves, or task lamp.

The semantic anchor should be a useful place for the robot to stand and observe the purposeful region, not the closest point to any single object.

## Duplication And Memory Rules

Semantic memory must avoid duplicating the same place.

Before creating a new named place:

1. Normalize label:
   - `living room`, `living_room`, `lounge` may be compatible
   - `dining`, `dining_area`, `eating_area` may be compatible
2. Compare evidence pose distance to existing places.
3. Compare anchor pose distance to existing places.
4. Compare source evidence overlap.
5. Compare semantic compatibility.

Merge if:

- label is same or compatible
- evidence clusters overlap or are close
- anchor poses are close enough, for example under `2.0m`
- source frames show the same area from similar/adjacent poses

Do not merge if:

- labels conflict strongly, for example `kitchen` versus `bedroom`
- evidence points are far apart
- the apartment likely has multiple instances, for example two bedrooms or two desk areas

If multiple instances exist, create indexed labels:

- `bedroom_1`
- `bedroom_2`
- `desk_area_1`
- `desk_area_2`

If new evidence is stronger than old evidence for the same place:

- keep the same `place_id`
- update the `anchor_pose` if the new anchor is safer, more central, or better observes the region
- append source evidence
- increase confidence if justified

Never delete old evidence immediately. Mark it as merged or superseded.

Duplication must be handled at three levels:

1. Pixel/object duplicate:
   - Same visual object appears in adjacent frames.
   - Merge into the same semantic evidence cluster if projected coordinates and labels agree.
2. Place duplicate:
   - Same kitchen/dining/living area observed from different positions.
   - Merge into one named place if evidence clusters are spatially and semantically compatible.
3. Anchor duplicate:
   - Multiple valid robot poses can observe the same place.
   - Keep the best anchor as primary and retain alternatives only if they provide meaningfully different visibility.

Do not create a new named place just because a new frame sees the same object from another angle.

Do create a separate named place when:

- two semantically similar areas are physically separate, such as two bedrooms
- two different purposeful zones share a large room, such as dining area and living room
- the LLM can explain the distinction using visible evidence and map context

## LLM Responsibilities

The LLM/VLM should:

- identify visible household semantic cues in RGB images
- group multiple cues into candidate places
- propose label hints and confidence
- explain why a place label is plausible
- decide whether evidence should create, update, merge, or reject named place memory
- avoid overconfident labels from weak evidence
- handle multiple possible labels by marking low-confidence provisional candidates

The LLM/VLM should not:

- invent ungrounded map coordinates
- directly choose robot navigation coordinates without deterministic snapping
- mutate frontier memory
- decide occupancy
- claim a named place is final if only weak evidence exists

The LLM/VLM should be explicitly told that semantic place labels are allowed to be provisional.

It is better to output:

```json
{
  "label_hint": "possible_kitchen",
  "confidence": 0.48,
  "reasoning_summary": "Cabinet-like surfaces are visible, but there is no sink, stove, or fridge evidence yet."
}
```

than to overstate:

```json
{
  "label_hint": "kitchen",
  "confidence": 0.95,
  "reasoning_summary": "This is definitely the kitchen."
}
```

## Prompt Contract For Semantic Evidence Extraction

Input to VLM per frame should include:

- RGB image
- optional depth visualization or RGB-D summary
- camera pose in map frame
- current navigation map image
- prior named places and semantic evidence memory

Ask for structured output:

```json
{
  "frame_id": "kf_011",
  "semantic_observations": [
    {
      "label_hint": "kitchen",
      "confidence": 0.74,
      "visual_cues": ["counter", "cabinet"],
      "pixel_regions": [
        {
          "description": "counter-like horizontal surface on left side",
          "image_position": "left_center",
          "bbox_xyxy": [20, 180, 260, 360],
          "representative_point_uv": [140, 270],
          "depth_hint": "mid"
        }
      ],
      "reasoning_summary": "Cabinet/counter-like surfaces indicate possible kitchen area."
    }
  ]
}
```

The VLM must identify multiple observations if multiple purposeful areas are visible in the same image.

The prompt should tell the VLM:

- Return pixel regions, not map coordinates.
- Prefer bounding boxes or representative pixels on visible objects that justify the label.
- Include multiple cues if a region has several purposeful objects.
- Use `unknown_semantic_area` when the area is visible but not labelable.
- Do not label empty open floor as kitchen/living/dining unless supporting objects are visible.
- Do not create a semantic observation from a frontier marker alone.
- Do not duplicate an existing named place unless there is new evidence that materially improves it.

## Prompt Contract For Semantic Consolidation

Input to consolidation LLM should include:

- final or current occupancy map image
- all semantic evidence clusters
- source frame thumbnails for each cluster
- robot trajectory
- current named place memory
- anchor reachability results

Output:

```json
{
  "place_updates": [
    {
      "action": "create",
      "target_place_id": null,
      "label": "kitchen",
      "source_anchor_id": "anchor_014",
      "confidence": 0.82,
      "evidence": [
        "counter and cabinets observed",
        "multiple frames agree on left-side kitchen-like area"
      ],
      "notes": "Use anchor pose near open floor facing cabinet/counter evidence."
    },
    {
      "action": "merge",
      "target_place_id": "place_dining_area_001",
      "label": "dining_area",
      "source_anchor_id": "anchor_020",
      "confidence": 0.78,
      "evidence": [
        "table and chair cluster visible in central open area"
      ],
      "notes": "New evidence improves anchor pose but refers to same dining area."
    }
  ]
}
```

Valid actions:

- `create`
- `update`
- `merge`
- `reject`
- `keep`

The consolidation prompt should tell the LLM:

- Named places are navigation destinations, not exact room polygons.
- A named place should have one best `anchor_pose` where the robot can stand.
- A named place can represent a sub-area inside a larger room, for example `desk_area` inside `living_room`.
- A named place can overlap another named place if the labels describe different useful purposes.
- Prefer stable labels with evidence over speculative labels.
- Merge duplicates aggressively when evidence describes the same place.
- Keep multiple instances only when there is strong spatial separation or label distinction.

## When To Run Semantic Processing

During normal exploration:

- after every full scan
- after arriving at a frontier
- optionally every N travel frames if camera images are useful

Do not stop frontier exploration just to inspect semantic candidates by default.

After exploration:

- run a final consolidation pass
- optionally request operator review of named places

Only navigate specifically for semantic inspection if:

- final semantic confidence is low
- the operator requests better semantic labels
- a target region has no good source frames
- the semantic inspection pose is close and does not derail exploration

## Integration With Existing Code

The existing frontier exploration flow should stay:

```text
scan -> update occupancy -> detect frontiers -> LLM chooses frontier -> Nav2/teleport moves -> scan
```

Semantic processing should attach to scan/keyframe creation:

```text
append keyframe -> enqueue semantic evidence extraction -> update semantic memory
```

Recommended modules:

- `xlerobot_playground/semantic_evidence.py`
- `xlerobot_playground/semantic_anchors.py`
- `xlerobot_playground/semantic_memory.py`
- `xlerobot_agent/semantic_prompts.py`

Avoid putting semantic evidence logic directly inside frontier detection methods.

The following functions/classes should not import semantic modules:

- occupancy-grid fusion
- frontier extraction
- frontier reachability checks
- frontier memory updates
- Nav2 goal execution

Semantic modules may import/read:

- keyframe records
- RGB image files or arrays
- depth image arrays
- camera intrinsics
- camera/world pose
- current occupancy grid snapshot
- free-space reachability helper

Semantic modules may write only:

- semantic evidence memory
- semantic anchor memory
- named place memory
- operator-review artifacts

Semantic modules must not write:

- occupancy grid cells
- frontier statuses
- active frontier
- visited frontier list
- failed frontier list
- Nav2 parameters

## Implementation Steps

### Step 1: Disable Old Free-Space Semantic Waypoints

Remove or gate behind a disabled flag:

- automatic free-space semantic waypoint generation
- blue waypoint queue
- `inspect_semantic_waypoint` as a normal planner action
- scheduling semantic waypoints before/after frontiers

Keep:

- named place output in final map
- named place rendering
- semantic update support if it does not interfere with frontier exploration

The old blue free-space semantic waypoint behavior should be considered deprecated.

If keeping code temporarily for debugging, it must be disabled by default behind an explicit flag such as:

```text
--experimental-free-space-semantic-waypoints
```

That flag should not be used in normal exploration runs.

### Step 2: Add Semantic Evidence Data Structures

Create:

- `SemanticEvidence`
- `PixelRegion`
- `SemanticAnchorCandidate`
- `NamedPlace`
- `SemanticMemory`

Semantic memory should store:

- raw evidence
- evidence clusters
- anchor candidates
- named places
- rejected/merged records for traceability

### Step 3: Add VLM Semantic Observation Call

After keyframe capture:

1. Build prompt with RGB image and context.
2. Ask VLM for semantic observations.
3. Parse JSON.
4. Validate fields.
5. Store raw observation trace.

The first version can use local Ollama multimodal models if available.

### Step 4: Project Pixel Evidence Into Map Coordinates

Implement deterministic projection:

```text
pixel + depth + camera intrinsics + camera pose -> 3D point -> map point -> 2D evidence pose
```

Fallback for coarse VLM outputs:

```text
image_position + depth statistics + camera ray sector -> approximate evidence pose
```

Do not let the LLM output final coordinates without projection.

### Step 5: Create Safe Anchors Near Evidence

For each evidence point or evidence cluster:

1. Search nearby known free cells.
2. Validate robot footprint clearance.
3. Validate known-free reachability.
4. Score line of sight to evidence.
5. Choose best anchor pose.
6. Set yaw to face evidence.

### Step 6: Cluster And Deduplicate

Cluster evidence before named-place updates.

Merge similar evidence by:

- label compatibility
- spatial proximity
- source frame relation
- object cue compatibility

Avoid duplicate named places.

### Step 7: LLM Consolidates Named Places

Periodically and at the end:

- provide evidence clusters
- provide anchor candidates
- provide map image
- provide source RGB images
- provide current named place memory

The LLM returns create/update/merge/reject decisions.

### Step 8: Operator Review

UI should show:

- named place labels
- evidence points
- anchor poses
- source image thumbnails
- confidence
- evidence notes

Operator can:

- accept label
- rename label
- delete named place
- move anchor pose
- mark evidence as wrong

## Coding Instructions For Future Implementation

Write clean, isolated code.

Do not bolt semantic logic into the frontier path.

Prefer this shape:

```text
xlerobot_playground/
  semantic_evidence.py       # data models, validation, VLM output parsing
  semantic_projection.py     # pixel/depth/camera pose -> map coordinates
  semantic_anchors.py        # safe reachable anchor selection
  semantic_memory.py         # clustering, dedupe, named place memory

xlerobot_agent/
  semantic_prompts.py        # VLM and consolidation prompts
```

The exploration backend should call semantic code only at keyframe boundaries:

```python
semantic_observations = semantic_extractor.extract(keyframe)
semantic_points = semantic_projector.project(semantic_observations, keyframe)
semantic_anchors = semantic_anchor_builder.build(semantic_points, occupancy_snapshot)
semantic_memory.update(semantic_anchors)
```

The frontier backend should not know why a place is called kitchen, dining area, or living room.

The semantic backend should not know how frontiers are selected.

All LLM/VLM outputs must be schema validated before use.

Any field that cannot be validated should be dropped with a traceable warning artifact, not silently accepted.

Use deterministic IDs:

- `sem_ev_000123`
- `sem_cluster_000045`
- `sem_anchor_000038`
- `place_kitchen_001`

Every stored semantic object should include:

- source frame ids
- source pixel regions or masks
- projected evidence pose
- anchor pose if available
- confidence
- evidence notes
- status
- timestamps or step indices

Never overwrite a named place without keeping the previous evidence trail.

## Failure Modes To Handle

The semantic layer must handle:

- invalid depth at the selected pixel
- object visible through glass or mirror
- object too close to robot
- evidence projected into occupied cells
- evidence projected outside known map
- no reachable anchor near evidence
- multiple possible labels for one visual cue
- duplicate observations from repeated scans
- conflicting labels from different frames
- low-confidence VLM output
- local model returning malformed JSON

Expected behavior:

- keep raw evidence when useful
- mark anchor as unavailable when no safe pose exists
- ask consolidation LLM to reject or keep provisional evidence
- never crash exploration because semantic interpretation failed
- never mutate occupancy/frontier state because semantic interpretation failed

## Testing Requirements

Add tests for:

- semantic evidence JSON validation
- duplicate evidence clustering
- evidence point projection math
- anchor pose snapping to nearby free space
- anchor rejection when no known-free reachable pose exists
- named place merge/update behavior
- semantic layer does not mutate frontier memory
- semantic layer does not mutate occupancy map unless explicitly allowed
- prompt includes RGB-D grounding instructions
- prompt forbids invented coordinates

## Quality Bar

Code should be modular.

Semantic code must not be hidden inside frontier methods.

All LLM outputs must be schema-validated.

All coordinates produced by LLM must be treated as hints unless grounded by RGB-D projection or snapped by deterministic code.

Every named place must have evidence, source frame ids, and a reachable anchor pose or an explicit unreachable status.

Every merge/update must preserve traceability.

The system must be able to explain:

- why a place exists
- which images support it
- where the evidence is
- where the robot should stand
- whether the anchor is reachable

## Summary

The semantic waypoint system should not sample free space.

It should discover meaningful places from visual evidence, ground that evidence through RGB-D into the map, and compute safe nearby navigation anchors.

Frontiers remain the mechanism for discovering space.

Semantic evidence and named places become the mechanism for understanding the apartment layout.
