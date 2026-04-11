# XLeRobot Agentic Exploration And 2D Mapping Implementation Plan

## Goal

Build a new exploration backend where:

- the robot builds an initial 2D occupancy map
- a deterministic mapping and viewpoint pipeline maintains exploration state
- an LLM chooses the next high-level exploration action
- Nav2 handles safe path planning and execution, eg. navigation
- the operator can inspect progress and review the resulting map

The two requirements that remain non-negotiable are:

- robust 2D occupancy map construction for navigation
- operator review of exploration progress and final map output
- llm as frontiere exploration agent
- nav2 as navigation low level module

## Product Definition

### Mission

Explore the accessible environment to construct a 2D map of the environment, specifically map of all the rooms and areas of interest for the robot. This map will be used as a predefined 2D navigation waypoint to areas of interest, such as kitchen, bathroom, bedroom, living room, desk area and so on. What the result would be:

- a 2D occupancy map
- region candidates and semantic evidence
- operator-reviewable exploration artifacts

### Exploration Control Split

The LLM is responsible for:

- maintaining a set of goals on what to explore and updating those goals properly
- starting from a given point after a complete turnaround scan happens (360 degrees)
- receiving the current 2D map composed so far, rendered images of the space, the robot position in that map, and the information needed to understand what is currently visible and what is still unexplored
- receiving deterministic frontier information, meaning map coordinates at the boundary between explored free space and unknown space
- understanding that frontier information is generated from the currently scanned RGB-D evidence and is not complete information about the whole apartment
- treating frontier information as boundary evidence, not as a command that every boundary should be explored
- using RGB visual input, the 2D map, robot map position, frontier information, and frontier memory to select specific useful regions to explore
- prioritizing frontier regions that likely expand robot-navigable floor space, such as room entrances, doors, corridors, open areas, and meaningful sensor-range-limit expansions
- deprioritizing boundaries that appear to be caused by furniture or clutter, such as the back or underside of a couch, table, cabinet, or shelf, unless there is evidence of traversable space beyond it
- selecting one frontier id to actively explore while the remaining useful frontier information stays in memory
- creating and updating frontier memory points in the same structured response as the frontier decision, including which points should be stored, prioritized, suppressed, or revalidated
- keeping frontier memory so the robot can return later to previously discovered but not yet explored useful frontiers
- choosing a waypoint to go back to if one area has finished exploring and there are still stored frontier points in memory that should be revisited
- exploring frontier points at the edge of what the RGB-D map can currently see, including when mapped observations reach the sensor limit of 10m and expansion beyond that boundary is still possible
- deciding when exploration is over by using a flag in its response
- generating labels and areas for things like kitchen, living room, bathroom and so on, and optionally sub-areas such as an office area or table eating area inside a larger room

The LLM is given:

- the current 2D map
- rendered image views of the space
- the robot position in the 2D map
- RGB-D-derived information that helps explain what the mapper currently sees
- frontier information, including boundary coordinates, evidence, reachability, and path cost
- frontier points currently in memory
- visited frontier points
- already explored areas

The LLM should operate on map-level frontier information and selected frontier ids, not raw RGB-D point targets. It should choose useful navigable-space exploration regions, not blindly select every boundary.

The deterministic path is responsible for:

- building and updating the 2D occupancy map
- updating the map after scans and after waypoint travel
- maintaining memory for active, stored, visited, failed, and completed frontier exploration points
- producing frontier information from occupancy-map boundaries
- preserving stable frontier ids and enforcing safety/validity on any LLM memory updates
- executing a full turnaround scan when needed and updating the map from that scan
- converting selected map-level frontier exploration points into valid navigation waypoints for Nav2
- using Nav2 to navigate the place safely
- keeping the remaining frontier points available for later exploration

### Frontier Memory

Frontier exploration points should be treated as persistent memory objects.

That memory should include:

- active frontier exploration point
- stored frontier exploration points not yet explored
- visited frontier exploration points
- failed frontier exploration points
- completed frontier exploration points
- return waypoints that let the robot go back and continue exploring another remembered frontier

When the LLM creates or updates frontier memory points:

- they must be checked against memory
- duplicates must be rejected
- already visited points must not be reintroduced as new frontiers
- already failed points must not be reintroduced as new frontiers unless explicitly revalidated by deterministic logic
- suppressed furniture/clutter boundary points should stay suppressed unless new visual or map evidence suggests traversable space

### Frontier Expansion At Sensor Range Limits

The system should treat the edge of the currently visible RGB-D map as a possible frontier when exploration can continue beyond it.

In particular:

- if the RGB-D sensing range reaches 10m
- and mapped observations extend to that 10m limit
- and the edge still suggests unexplored accessible space

then that edge should also be considered a frontier exploration point candidate for expansion.

### When Its Over

LLM is the one deciding when its over. It can also decide to return to previous points in order to explore other avenues. The LLM decides when the exploration is over by using a flag in its response.

### LLM Mapping

- During this whole process LLM can also generate labels and areas for this labels. LLM is instructed to generate labels for things like: new areas such as kitchen, living room, bathroom and so on. It can also generate areas inside areas for instance inside living room it can detect the office area or the table eating area.

### Nav2 and navigation

- Nav2 and sensor data are responsible for proper navigation, which can be things like navigating through open door, making sure to avoid obstacles such as low tables, or even if needed being able to navigate corners.
