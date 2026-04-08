# XLeRobot Agent Architecture v1

Note: this is the simplified implementation spec.

## Goal
Build one clear agent loop for XLeRobot that can:

- wake up from a voice trigger
- turn speech into a normalized instruction
- look at scene context from images and 3D memory
- discover likely places such as `kitchen`
- break the instruction into subgoals
- choose skills for each subgoal with prompting
- execute either:
  - skill-only navigation and manipulation
  - delegated Nav2-based navigation plus skills

## Core Idea
Everything is a skill at the planner level.

The agent does not plan in raw motion.
It plans in simple subgoals and selects skills from the registered skill set.

The agent may also use a very small set of non-actuation tools when skills alone are not enough:

- navigation / mapping tools
- perception grounding tools
- bounded code execution for analysis

The runtime shape is:

`wake word -> speech -> normalized instruction -> place discovery -> subgoal planning -> skill selection -> execution -> verification -> memory update`

## Inputs
The agent can use:

- text instruction
- spoken instruction after wake-word activation
- head camera context
  - RGB
  - depth
  - point cloud
  - IMU-derived context
- left arm camera RGB
- right arm camera RGB
- semantic memory
- spatial / 3D memory
- current skill registry
- current execution mode

## Voice Path
Voice is a front door to the same planner.

Required behavior:

- listen for a wake word such as `hey xlerobot`
- ignore speech without the wake word
- normalize or translate the spoken command into plain planner text
- feed the normalized text into the same agent path as typed commands

Voice is not a separate planner.

## World Understanding
The agent should keep one simple world context object.

That context should contain:

- current task
- current pose
- localization confidence
- visible objects
- visible landmarks
- image descriptions or image-derived summaries
- semantic memory summary
- spatial memory summary
- discovered places
- active resource locks
- recent execution history

The context metadata should also be able to carry perception annotations when available:

- segmented object instances
- 2D bounding boxes
- mask ids
- 3D centroids / anchors
- overlay text for operator-facing UI
- waypoint hints derived from depth or point-cloud geometry

### Place Discovery
The agent should be able to infer likely places from visual and memory evidence.

Examples:

- fridge / oven / sink -> likely `kitchen`
- sofa / tv -> likely `living_room`
- bed / pillow -> likely `bedroom`

This does not need a heavy place-classification system in v1.
It only needs a prompt or heuristic layer that can assign:

- place name
- confidence
- evidence

## Perception Stack
Do not overload the planner with many perception APIs.

Use one compact perception layer that can:

- refresh a scene understanding snapshot from RGB-D and point-cloud context
- ground a text target into a segmented object with a 3D anchor
- derive a waypoint or approach pose from that object anchor

This gives the planner an alternative to:

- relying only on a full precomputed map
- relying only on predefined named places
- relying only on end-to-end VLA actuation

### Perception Tool Surface
Keep the v1 tool surface small:

- `perceive_scene`
- `ground_object_3d`
- `set_waypoint_from_object`

These are analysis / world-building tools.
They do not directly actuate the robot.

Their job is to feed:

- `go_to_pose`
- navigation skills
- alignment skills
- manipulation skills

### Annotation Contract
Each perception snapshot should be able to emit annotations like:

- `label`
- `confidence`
- `bbox_2d`
- `mask_id`
- `centroid_3d`
- `depth_m`
- `overlay_text`
- `waypoint_hint`

The UI can use `overlay_text` and the 2D/3D anchor data to render object overlays for the operator.

## Skills
The planner may only choose from registered skills.

Skills include:

- manipulation skills
- navigation skills
- search skills
- alignment skills
- recovery skills

Example manipulation skills:

- `open_fridge`
- `grab_bread_from_table`
- `unpack_groceries_from_bag`
- `clean_pens_on_desk`

Example navigation skills:

- `navigate_to_region`
- `approach_target`
- `move_to_viewpoint`
- `align_for_skill`
- `retreat_from_target`

Each skill must define:

- `skill_id`
- `skill_type`
- language description
- executor binding
- preconditions
- required observations
- required resources
- expected postcondition
- retry cap

## Execution Modes
Navigation-related skills support two execution modes selected at startup:

### 1. `vla_navigation_skills`
Use learned navigation skills directly.

This means navigation remains fully skill-based.

### 2. `delegated_navigation_module`
Use a delegated navigation backend under the same skill interface.

Supported delegated backends:

- `progressive_map`
- `global_map`

This means the planner still chooses navigation skills, but their executor forwards to Nav2-style navigation.

The planner must not change when this switch changes.

## Planning
The planner should stay simple.

It has three jobs:

### 1. Normalize the instruction
Turn raw text or voice into one clean instruction string.

### 2. Break the instruction into subgoals
Examples:

Instruction:
`go to the kitchen and open the fridge`

Subgoals:

- `go to the kitchen`
- `find the fridge`
- `align with the fridge`
- `open the fridge`

### 3. Pick one skill for each subgoal
For each subgoal, score the feasible skills with prompting.

The prompt should use:

- subgoal text
- current world context
- discovered places
- image summaries
- semantic / spatial memory
- available skills
- active execution mode

The prompt should return:

- `skill_id`
- usefulness score
- executability score
- combined score
- reasoning

The implementation may keep:

`combined_score = usefulness * executability`

but the source of those values is the prompt, not separate hand-written model classes.

## Tool Strategy
Do not make everything a skill.

Use three categories:

### 1. Actuation Skills
These move the robot or manipulate the world.

Examples:

- `open_fridge`
- `grab_bread_from_table`
- `navigate_to_region`
- `align_for_skill`

### 2. High-Level Task Tools
These interact with delegated services such as Nav2 / mapping.

Examples:

- `go_to_pose`
- `get_map`
- `explore`
- `create_map`

### 3. Perception Tools
These build scene knowledge for the planner and for the operator UI.

Examples:

- `perceive_scene`
- `ground_object_3d`
- `set_waypoint_from_object`

This keeps the agent coherent:

- skills act
- tools observe, structure, or delegate
- the planner chooses the smallest useful step

## Deterministic Gate
Before a skill is scored, reject it if:

- the skill is disabled
- executor binding is missing
- required resources are locked
- required observations are missing
- localization is below the skill threshold
- preconditions are not satisfied

This gate must stay outside the prompt.

## Readiness
The execution layer must still distinguish:

- `navigation_ready_pose`
- `perception_ready_pose`
- `skill_ready_pose`

The planner may select a skill, but manipulation must only run when readiness is sufficient.

## Runtime Loop
1. Wait for wake word if voice mode is active.
2. Normalize the command.
3. Build the current world context from images, memory, and robot state.
4. Run the Visual Differencing Module once on the current observations to produce:
   - a scene summary
   - task-relevant visual attributes
   - an initial completion guess
5. Refresh scene understanding if the current task needs perception grounding.
6. Discover likely places.
7. Break the instruction into subgoals.
8. For each subgoal:
   - gate infeasible skills
   - expose compact navigation / perception tools
   - prompt-rank feasible skills
   - select the best skill
   - or select a tool that improves world understanding or delegates navigation
   - execute it
   - run the Visual Differencing Module again on the previous and current observations
   - append the visual delta summary into world memory and the operator report
   - verify the result
   - update memory and history
9. Stop when all subgoals are completed or execution fails.

Example:

`find the fridge and open it`

- `perceive_scene`
- `ground_object_3d("fridge")`
- `set_waypoint_from_object("fridge")`
- `go_to_pose(...)` or `align_for_skill`
- `open_fridge`

## Visual Differencing Module
The Visual Differencing Module is an observation module, not a controller.

It should:

- summarize the initial scene in natural language
- extract task-relevant visual attributes
- compare previous and current observations after each action
- describe what changed and what did not change
- provide a lightweight completion signal for the critic and operator UI

It should not:

- directly command skills
- bypass the planner
- replace segmentation or 3D grounding backends

The intended use is:

- perception tools produce structured scene state
- the Visual Differencing Module converts that structured state into compact natural-language evidence
- the critic and planner consume that evidence for retry, replay, cancel, or replan decisions

## Safety
Safety must remain outside the prompt and outside learned skills.

The safety layer owns:

- hard stop
- unsafe motion veto
- human proximity rules
- collision veto
- failure escalation

## Initial Testing Targets
The first tests should prove:

- wake word blocks accidental commands
- voice and typed commands go through the same planner
- place discovery can infer `kitchen` from fridge evidence
- subgoal decomposition works on simple household tasks
- navigation skills can run in both:
  - `vla_navigation_skills`
  - `delegated_navigation_module`
- delegated mode can swap:
  - `progressive_map`
  - `global_map`
- perception tools can:
  - emit object overlays with 2D + 3D anchors
  - ground a queried target into a segment / anchor
  - derive a waypoint hint from depth or point-cloud context
- planner always chooses from registered skills only
