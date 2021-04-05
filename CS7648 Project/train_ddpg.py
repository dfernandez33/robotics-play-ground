from robomaster import robot
from ddpg import DDPG
from noise import OUNoise
import torch
import time
import argparse
from robot_manager import RobotManager
from utils import calculate_reward


def train(agent: DDPG, manager: RobotManager, epochs: int, trajectory_length: int,
          noise_model: OUNoise, batch_size: int):

    for epoch in range(epochs):
        cum_reward = 0
        noise_model.reset()
        manager.reset_arm()
        state = manager.get_state()
        print("Starting Epoch: {}".format(epoch))

        for step in range(trajectory_length):
            if manager.battery_level < 10:
                manager.ep_robot.play_sound(robot.SOUND_ID_SCANNING).wait_for_completed()
                print("Stopped training due to low battery. Model has been saved to: {}".format(agent.save_path))
                return agent
            action = agent.get_action(state)
            action = noise.get_action(action.detach().numpy())
            terminal = manager.execute_action(action)
            next_state = manager.get_state()
            reward = calculate_reward(terminal, next_state, action)
            cum_reward += reward.item()
            agent.memory.push(state.float(), torch.from_numpy(action).float(), reward.float(), next_state.float(), terminal)
            if len(agent.memory) >= batch_size and step % 4 == 0:
                agent.update(batch_size)
            if terminal:
                break
            state = next_state
        print("Finished Epoch {} with total reward: {}".format(epoch, cum_reward))

    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', help="where to save the model", type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--num-inputs', help="number of input parameters to the networks", type=int, default=10)
    parser.add_argument('--actor-outputs', help="number of actions for the actor network", type=int, default=10)
    parser.add_argument('--hidden-size', help="number of nodes in each hidden layer", type=int, default=256)
    parser.add_argument('--soft-update-weight', type=float, default=.01)
    parser.add_argument('--discount-factor', type=float, default=.95)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--max-memory', help="maximum capacity for replay buffer", type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trajectory-length', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cuda', help="Set to True in order to use GPU", type=bool, default=False)

    args = parser.parse_args()

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type='ap', proto_type='udp')
    manager = RobotManager(ep_robot)

    time.sleep(1)

    agent = DDPG(args.num_inputs, args.actor_outputs, args.hidden_size, args.soft_update_weight,
                 args.discount_factor, args.actor_lr, args.critic_lr, args.max_memory, args.save_path, args.checkpoint_path, args.cuda)
    noise = OUNoise(args.actor_outputs)

    learned_agent = train(agent, manager, args.epochs, args.trajectory_length, noise, args.batch_size)
    agent.save_agent()

    ep_robot.close()
