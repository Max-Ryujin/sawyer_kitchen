## Branches
- moving_minimal is the simplest possible setup for only moving the cup that works.



## Files

Here are all important files including their most important functions:

- test_env.py: contines the code to run the handcrafted policy a single time to generate goal observations and videos for debugging policy.
    - make_task_space_action() creates valid actions from targets (does normalisation and clipping).
    - moving_policy() the handcrafted policy to move the cup to the goal.
    - collect_policy_episode() runs the policy in the environment and collects observations and videos.
- parallel_test.py: contains the code to generate datasets using multiple workers in parallel.
    - make_task_space_action() creates valid actions from targets (does normalisation and clipping).
    - moving_policy() the handcrafted policy to move the cup to the goal.
    - run_single_episode() worker function to generate a episode.
    - collect_moving_policy_dataset() main function to generate dataset using multiple workers.
- train_CRL.py: contains the training code for all RL agents.
    - evaluate_agent() evaluates the agent during checkpoints on a fixed test case and a set of validation episodes.
    - main() main training loop.
- env.py: contains the enviroment.
    - step() steps the environment given an action and does inverse kinematics and denormalisation.
    - reset() resets the environment and can place the cups randomly.
    - _get_observation() gets the observation vecor containing currently only one cup and the gripper.
    - check_moving_success() checks if the cup has been moved to the goal position.





## Changes 
Here are the changes that I did to get the agents to work:
- Changed the observation space to only contain one cup and the gripper.
    - The second cup that is not moved was used by the contrstive critic and prevented learning.
- removed rotation from action space.
    - quaternions are hard to learn for RL agents and not needed for moving the cup.
- added back normalisation
- fixed a bug in the sucess check for moving the cup.


### commit  8033948
- moved policy code from test_env.py and parallel_test.py to policies.py
- added rotation around z and xy axis to the action space of the moving policy
- made the action range consistent between 0 and 1 for all dimensions to have similar scaling
- (need to add rotations to the policy to be able to generate a new dataset)

### commit 9b1a4e3
- added relative rotations to the action space in the policies.
- Since the rotations are relative to a fixed quaternion and applied one after the other, it is quite hard to calculate the correct values for the action space in the observations.
- Still need to implement a function that takes the current quaternion and extracts the rot_z and rot_xy values relative to a set of fixed quaternions.
- Need to remember to update the GOAL_OBS once I am able to generate correct observations with rotations.


### notes on rotation representation
- currently using two values rot_z and rot_xy to represent rotation around the z axis and rotation around the x and y axis.
- the advantage is that this representation is quite compact and easy to normalise between 0 and 1.
- the disadvantage is that it is quite hard to convert between quaternions and this representation. I need to do a gridserach over possible quaternions to find the best matching rot_z and rot_xy values. This is not very efficient but works and I am concerned about gradinents as well.
- another option would be to use euler angles, but they have singularities and other issues.
- In the long term it might be worth checking out https://arxiv.org/abs/1812.07035 for better rotation representations for RL.

### commit ed6be57
- changed the action and observation space to only use one scalar for roation to make puring possible again.
- I am currently choosing a random rotation at the start of the episode.
- It seems that the critic uses the rotation information in CRL to differenciate between the states like it did with two cups. 
- I will test to use a set of preselected angles to see if that improves performance.