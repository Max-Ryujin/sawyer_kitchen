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