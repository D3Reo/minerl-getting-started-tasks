import torch.nn
from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import gym
import torch.nn.functional as F
import minerl
from minerl.herobraine.wrappers.video_recording_wrapper import VideoRecordingWrapper
from minerl.data.buffered_batch_iter import BufferedBatchIter

RNG = np.random.default_rng(42)

DATA_DIR = "/home/d3reo/data/minerl/"
EPOCHS = 5
BATCH_SIZE = 32
VIDEO_DIR = "./videos"

NO_CAM_HIERARCHY_MAP = []
CAM_HIERARCHY_MAP = []
HIERARCHY_MAP = ["attack", "jump", "look_left", "look_right", "forward",  "sprint", "look_up", "look_down", "right", "left", "back", "sneak"]

LOOK_AROUND_MAP = {
    "look_left": [0., -10.],
    "look_right": [0., 10.],
    "look_down": [10., 0.],
    "look_up": [-10., 0.]
}


class ConvNet(nn.Module):
    """
    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.LazyLinear(2*output_dim)
        self.linear2 = nn.LazyLinear(output_dim)
        self.dropout = nn.Dropout(0.25)
        _ = input_shape

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = F.relu(self.conv1(observations))
        observations = self.dropout(observations)
        observations = F.relu(self.conv2(observations))
        observations = self.dropout(observations)
        observations = F.relu(self.conv3(observations))
        observations = self.dropout(observations)
        observations = F.avg_pool2d(observations, 4)
        observations = observations.flatten(start_dim=1)
        observations = F.relu(self.linear1(observations))
        observations = self.dropout(observations)
        observations = self.linear2(observations)
        return observations


def agent_action_to_environment(noop_action, agent_action):
    """
    Turn an agent action (an integer) into an environment action.
    This should match `environment_action_batch_to_agent_actions`,
    e.g. if attack=1 action was mapped to agent_action=0, then agent_action=0
    should be mapped back to attack=1.

    noop_action is a MineRL action that does nothing. You may want to
    use this as a template for the action you return.
    """
    possible_actions = HIERARCHY_MAP + ["noop"]
    assert agent_action < len(possible_actions)
    done_action = possible_actions[agent_action]

    if done_action != "noop":
        noop_action.update(
            {done_action: 1} if done_action[:4] != "look" else
            {"camera": LOOK_AROUND_MAP[done_action]})

    return noop_action


def environment_action_batch_to_agent_actions(dataset_actions, actions_dict):
    """
    Turn a batch of actions from environment (from BufferedBatchIterator) to a numpy
    array of agent actions.

    Agent actions _have to_ start from 0 and go up from there!

    For MineRLTreechop, you want to have actions for the following at the very least:
    - Forward movement
    - Jumping
    - Turning camera left, right, up and down
    - Attack

    For example, you could have seven agent actions that mean following:
    0 = forward
    1 = jump
    2 = turn camera left
    3 = turn camera right
    4 = turn camera up
    5 = turn camera down
    6 = attack

    This should match `agent_action_to_environment`, by converting dictionary
    actions into individual integers.

    If dataset action (dict) does not have a mapping to agent action (int),
    then set it "-1"
    """
    dataset_actions = dataset_actions.copy()

    # There are dummy dimensions of shape one
    camera_data = dataset_actions.pop('camera')
    batch_size = len(camera_data)
    actions = np.zeros((batch_size,), dtype=int)

    down_right_dict = {
        key: value
        for key, value in zip(
            ['look_down', 'look_right'],
            (camera_data >= 5).T
        )
    }

    up_left_dict = {
        key: value
        for key, value in zip(
            ['look_up', 'look_left'],
            (camera_data <= -5).T
        )
    }

    dataset_actions.update(down_right_dict)
    dataset_actions.update(up_left_dict)

    raw_actions_tensor = np.array(list(zip(
        *[
            dataset_actions[key]
            for key in HIERARCHY_MAP
        ]
    )), dtype=np.float32)
    assert raw_actions_tensor.shape[0] == camera_data.shape[0]

    noop_column = (raw_actions_tensor.sum(axis=-1, keepdims=True) == 0).astype(np.float32)

    actions_tensor = np.hstack([raw_actions_tensor, noop_column])
    assert actions_tensor.shape == (camera_data.shape[0], len(list(actions_dict)) + 4)

    actions += actions_tensor.argmax(axis=-1)
    return actions


def train(actions_dict):
    env_name = "MineRLTreechop-v0"
    data_pipeline = minerl.data.make(env_name, DATA_DIR)
    iterator = BufferedBatchIter(data_pipeline)

    number_of_actions = len(list(actions_dict)) + 4  # +1 for the noop action -1 +4 for camera to left right up down
    print(f"Number of actions is {number_of_actions} ({len(list(actions_dict))} + 4)")

    network = ConvNet((3, 64, 64), number_of_actions)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
    loss_function = torch.nn.CrossEntropyLoss()

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(iterator.buffered_batch_iter(
            num_epochs=EPOCHS, batch_size=BATCH_SIZE)):
        # We only use camera observations here
        obs = dataset_obs["pov"].astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations, otherwise the neural network will get spooked
        obs /= 255.0

        # Turn dataset actions into agent actions
        actions = environment_action_batch_to_agent_actions(dataset_actions, actions_dict)
        assert actions.shape == (obs.shape[0],),\
            "Array from environment_action_batch_to_agent_actions should be of shape {}".format((obs.shape[0],))

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

        pred = network(torch.from_numpy(obs))
        exp = torch.from_numpy(actions)
        loss = loss_function(pred, exp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep track of how training is going by printing out the loss
        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()

    # Store the network
    th.save(network, "model_behavioural_cloning.pth")


def enjoy(env):
    # Load up the trained network
    network = th.load("model_behavioural_cloning.pth")

    # Play 10 games with the model
    for game_i in range(10):
        obs = env.reset()
        done = False
        reward_sum = 0
        while not done:
            povs = torch.from_numpy(np.array([obs["pov"].astype(np.float32).transpose(2, 0, 1) / 255]))
            logits = network(povs)

            # Turn logits into probabilities
            probabilities = th.softmax(logits, dim=1)[0]  # NB removing batch dimension!!
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()

            agent_action = RNG.choice(len(probabilities), p=probabilities)
            # TODO improvement by "Sample action based on probabilities"

            noop_action = env.action_space.noop()
            environment_action = agent_action_to_environment(noop_action, agent_action)

            obs, reward, done, info = env.step(environment_action)
            reward_sum += reward
        print("Game {}, total reward {}".format(game_i, reward_sum))

    env.close()


if __name__ == "__main__":
    ENV = VideoRecordingWrapper(gym.make('MineRLTreechop-v0'), video_directory=VIDEO_DIR)

    # First train the model... comment if not needed
    train(ENV.action_space)
    # ... then play it on the environment to see how it does
    enjoy(ENV)
