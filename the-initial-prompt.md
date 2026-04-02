I want to create an agent that is capable of operating my XleRobot 0.4.0 robot. I'm going to describe the overall architecture of what I want to implement and we iterate over the architecture a bit up until we reach a point where implementation can be done.

The agent has access to the robot capabilities, what the SO arm 101 can do. This is exposed in form of skills, the robot is pretrained on a variaty of low level locomotory skills with its arms which some examples are:
- Open the fridge door.
- Grab the bread from the table.
- Unpack groceries from the bag.
- Clean the pens on the desk.
- Grab plastic bottle and put it in the cart tray.
- Recycle plastic from the tray into the trash.

And so on. The LLM is the center of the operation. It is able to get information about the environment, about the robot possibilities and act. The LLM is also given access to the current user instruction: eg. what to do. But this instruction as we will see in the upcoming examples, can range from simple ones like above to really complicated ones that integrate navigation, breaking apart into multiple skill actions, weighting value functions of those actions and deciding which combinations work best and then execute.

The LLM generally is able to break task into sub-task but here it is important for the agent to understand the robot operating capabilities and the state of the environment. This are critical.

The Agent architecture has to take into account that it not only has to break down the problem into subproblem and possibly use existing skills or navigations to solve them. It also has to be able to weight in their sequence, combine them in order to solve the problem (possibly through some kind of affordance function or prompt it can use) and then execute something that might have high changes to work in current environment. It also has to weight in the probability that each skill will work out well.  So these are two different probabilities that the Agent has to figure out: the probability a given skill will amplify the success of the goal and the probability the skill will succed. It operates on the combination of these skills and environment cues including, and most importantly 3D representation and navigation as well as 3D memory.

What agent will have access to:
- LLM for inference including a specialized LLM for 3D understanding.
- Gemini 2 camera information: RGB color map from the head area - Depth map per-pixel distance - Point cloud access from Orbec SDK - IR image, integrated IMU. I am not saying the agent will be given access to all that but that is available when constructing, creating the 3D representations (with ORbec sdk) and navigation module. Including on the part of obstacle detection.
- 2 cameras, one on the left arm the other on the right with RGB simple image feed for executing VLA actions.
- Access to skills and vla models to trigger the skills one, where the 2 arms will perform the task together with other parts such as moving the base or the head and so on. THe orbec camera rgb can be used as well. 3 camera feed.

Important aspect:
THe agent architecture doesn't incorporate just skill usage, objuective break down and so on. It incorporates an additional module which is 3D navigation and path planning. This is an important part of the architecture because the agent can only activate certain skills if the position allows it. Eg. the agent cannot activate open fridge if the fridge is in the kitchen and the agent is in the living room. For this to work the 3D planning module has the following approximate architecture

Here is the proposed architecture, please feel free to debate it and let me know if this would work:
