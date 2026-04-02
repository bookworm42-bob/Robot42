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
4. Discover likely places.
5. Break the instruction into subgoals.
6. For each subgoal:
   - gate infeasible skills
   - prompt-rank feasible skills
   - select the best skill
   - execute it
   - verify the result
   - update memory and history
7. Stop when all subgoals are completed or execution fails.

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
- planner always chooses from registered skills only
