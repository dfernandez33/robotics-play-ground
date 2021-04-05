import torch


def calculate_reward(terminal, state, action):
    distance_to_target = torch.linalg.norm(state[0, :3]).item()
    rotation_penalty = 0
    if abs(action[0][2]) > 180:
        # add rotation penalty if abs(rotation) > 180 since in that case it should've just turned less
        # in the opposite direction
        rotation_penalty = abs(action[0][2])
    if terminal:
        if -55 <= state[0, 0].item() <= 0 and -20 <= state[0, 1].item() <= 20 and -40 <= state[0, 2] <= 40:
            return torch.tensor([[50000.0 - rotation_penalty]])  # big reward for successfully grabbing ball
        else:
            # reward for closing the gripper at the wrong location is scaled by how far you were from the target
            return torch.tensor([[-10*distance_to_target - rotation_penalty]])
    else:
        return torch.tensor([[-distance_to_target - rotation_penalty]])