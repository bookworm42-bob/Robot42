# XLeRobot Agent Runtime

This repo now contains a simplified agent runtime.

## Current Shape

The runtime does four things:

1. accept text or wake-word voice commands
2. build simple world context with images and memory summaries
3. discover places and break the goal into subgoals
4. choose and execute registered skills for each subgoal

## Main Files

- [xlerobot_agent/runtime.py](/Users/alin/multido/xlerobot_agent/runtime.py)
  - simple top-level runtime
- [xlerobot_agent/scoring.py](/Users/alin/multido/xlerobot_agent/scoring.py)
  - prompt planner, place discovery, subgoal planning, skill scoring
- [xlerobot_agent/voice.py](/Users/alin/multido/xlerobot_agent/voice.py)
  - wake word and mock voice pipeline
- [xlerobot_agent/integration.py](/Users/alin/multido/xlerobot_agent/integration.py)
  - startup switch between skill-only navigation and delegated navigation
- [multido_xlerobot/interface.py](/Users/alin/multido/multido_xlerobot/interface.py)
  - XLeRobot fork integration boundary

## What Is Real

- skill registry
- prompt-shaped planning flow
- subgoal decomposition
- place discovery
- navigation mode switching
- dry-run execution
- wake-word-triggered mock voice app

## What Is Still Mocked

- real LLM prompt calls
- real microphone / ASR
- real perception pipelines from live images
- real postcondition verification
- real hardware-backed skill executors

## Examples

- [use_xlerobot_agent.py](/Users/alin/multido/examples/use_xlerobot_agent.py)
- [run_mock_voice_agent.py](/Users/alin/multido/examples/run_mock_voice_agent.py)
