import rl1
import torch
import copy
import torch.optim as optim
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn

class DQNTrainer:

    def __init__(self, pipeline):

        self.pipeline = pipeline

        self.device = pipeline.device

        self.policy_net = pipeline.policy.to(self.device)

        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=rl1.LEARNING_RATE
        )

        self.loss_fn = nn.SmoothL1Loss()

        self.memory = rl1.ReplayBuffer(rl1.BUFFER_SIZE)

        self.epsilon = rl1.EPS_START

        self.update_counter = 0


    def decay_epsilon(self):

        self.epsilon = max(
            rl1.EPS_END,
            self.epsilon * rl1.EPS_DECAY
        )


    def select_action(self, state):

        action, uncertainty = self.pipeline.select_action(state)

        explore = False

        if random.random() < self.epsilon:
            explore = True

        if uncertainty > rl1.UNCERTAINTY_THRESHOLD:
            explore = True

        if explore:

            action = random.randint(
                0,
                len(rl1.ACTIONS) - 1
            )

        return action, uncertainty


    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )


    def train_step(self):

        if len(self.memory) < rl1.BATCH_SIZE:
            return None

        batch = self.memory.sample(rl1.BATCH_SIZE)

        (
            emotions,
            risks,
            risk_vectors,
            uncertainties,
            embeddings,

            actions,
            rewards,

            next_emotions,
            next_risks,
            next_risk_vectors,
            next_uncertainties,
            next_embeddings,

            dones

        ) = rl1.extract_tensors(
            batch,
            self.device,
            self.pipeline.scaler
        )

        self.policy_net.train()


        current_q = self.policy_net(

            emotions,
            risks,
            risk_vectors,
            uncertainties,
            embeddings

        )

        current_q = current_q.gather(

            1,

            actions.unsqueeze(1)

        ).squeeze(1)

        with torch.no_grad():


            next_actions = self.policy_net(

                next_emotions,
                next_risks,
                next_risk_vectors,
                next_uncertainties,
                next_embeddings

            ).argmax(dim=1)

            next_target_q = self.target_net(

                next_emotions,
                next_risks,
                next_risk_vectors,
                next_uncertainties,
                next_embeddings

            )

            next_q = next_target_q.gather(

                1,

                next_actions.unsqueeze(1)

            ).squeeze(1)

            target_q = rewards + (

                (1 - dones)

                * rl1.GAMMA

                * next_q

            )


        loss = self.loss_fn(

            current_q,

            target_q

        )


        self.optimizer.zero_grad()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            self.policy_net.parameters(),

            max_norm=1.0

        )

        self.optimizer.step()


        self.update_counter += 1

        if self.update_counter % rl1.TARGET_UPDATE == 0:

            self.target_net.load_state_dict(

                self.policy_net.state_dict()

            )


        self.decay_epsilon()


        stats = {

            "loss": loss.item(),

            "epsilon": self.epsilon,

            "buffer_size": len(self.memory),

            "mean_reward": rewards.mean().item(),

            "mean_q": current_q.mean().item()

        }

        return stats




    def update_target_network(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()


    def soft_update_target_network(self, tau=0.005):

        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):

            target_param.data.copy_(

                tau * policy_param.data +

                (1.0 - tau) * target_param.data

            )


    def save_checkpoint(
        self,
        filename=rl1.CHECKPOINT_PATH,
        episode=0
    ):

        checkpoint = {

            "episode": episode,

            "policy_state_dict":
                self.policy_net.state_dict(),

            "target_state_dict":
                self.target_net.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "epsilon":
                self.epsilon,

            "update_counter":
                self.update_counter

        }

        torch.save(
            checkpoint,
            filename
        )

        print(f"\nCheckpoint saved to {filename}")


    def load_checkpoint(
        self,
        filename=rl1.CHECKPOINT_PATH
    ):

        checkpoint = torch.load(

            filename,

            map_location=self.device

        )

        self.policy_net.load_state_dict(

            checkpoint["policy_state_dict"]

        )

        self.target_net.load_state_dict(

            checkpoint["target_state_dict"]

        )

        self.optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

        self.epsilon = checkpoint.get(

            "epsilon",

            rl1.EPS_START

        )

        self.update_counter = checkpoint.get(

            "update_counter",

            0

        )

        print(f"\nCheckpoint loaded from {filename}")

        return checkpoint.get("episode", 0)


    def train_mode(self):

        self.policy_net.train()

    def eval_mode(self):

        self.policy_net.eval()


    def get_learning_rate(self):

        return self.optimizer.param_groups[0]["lr"]


    def replay_size(self):

        return len(self.memory)


    def reset_memory(self):

        self.memory = rl1.ReplayBuffer(rl1.BUFFER_SIZE)


    def print_training_status(
        self,
        episode,
        step,
        reward,
        stats
    ):
       
        if stats is None:

            print(
                f"[Episode {episode:03d}] "
                f"[Step {step:02d}] "
                f"Reward={reward:.3f} "
                f"| Filling Replay Buffer "
                f"({len(self.memory)}/{rl1.BATCH_SIZE})"
            )

            return

        print(

            f"[Episode {episode:03d}] "

            f"[Step {step:02d}] "

            f"Reward={reward:.3f} "

            f"| Loss={stats['loss']:.4f} "

            f"| ε={stats['epsilon']:.3f} "

            f"| Q={stats['mean_q']:.3f} "

            f"| Buffer={stats['buffer_size']}"

        )







    @torch.no_grad()
    def evaluate_policy(self, state):
        

        self.policy_net.eval()

        action, uncertainty = self.pipeline.select_action(state)

        return action, uncertainty


    @torch.no_grad()
    def compute_q_values(self, state):
        

        emotion = [state["emotion"].get(k, 0.0) for k in rl1.EMOTION_ORDER]

        features = np.concatenate([
            np.array(emotion),
            [state["risk"]],
            state["risk_vector"],
            [state["risk_uncertainty"]],
            state["embedding"]
        ]).reshape(1, -1)

        features = self.pipeline.scaler.transform(features)[0]

        emotions = torch.FloatTensor(features[:6]).unsqueeze(0).to(self.device)
        risks = torch.FloatTensor([features[6]]).unsqueeze(0).to(self.device)
        risk_vectors = torch.FloatTensor(features[7:16]).unsqueeze(0).to(self.device)
        uncertainties = torch.FloatTensor([features[16]]).unsqueeze(0).to(self.device)
        embeddings = torch.FloatTensor(features[17:]).unsqueeze(0).to(self.device)

        q_values = self.policy_net(
            emotions,
            risks,
            risk_vectors,
            uncertainties,
            embeddings
        )

        return q_values.squeeze(0).cpu().numpy()


    @torch.no_grad()
    def predict_action(self, state):
        

        q_values = self.compute_q_values(state)

        action = int(np.argmax(q_values))

        return action, ACTIONS[action]


    def initialize_episode_stats(self):

        return {
            "episode_reward": 0.0,
            "losses": [],
            "actions": [],
            "risks": [],
            "uncertainties": []
        }

    def update_episode_stats(
        self,
        stats,
        reward,
        action,
        risk,
        uncertainty,
        train_stats=None
    ):
        

        stats["episode_reward"] += reward

        stats["actions"].append(action)

        stats["risks"].append(risk)

        stats["uncertainties"].append(uncertainty)

        if train_stats is not None:
            stats["losses"].append(train_stats["loss"])

        return stats


    def summarize_episode(
        self,
        episode,
        stats
    ):
        

        avg_loss = (
            np.mean(stats["losses"])
            if len(stats["losses"]) > 0
            else 0.0
        )

        avg_risk = (
            np.mean(stats["risks"])
            if len(stats["risks"]) > 0
            else 0.0
        )

        avg_uncertainty = (
            np.mean(stats["uncertainties"])
            if len(stats["uncertainties"]) > 0
            else 0.0
        )

        print("\n" + "=" * 70)
        print(f"Episode {episode} Summary")
        print("=" * 70)

        print(f"Total Reward      : {stats['episode_reward']:.3f}")
        print(f"Average Loss      : {avg_loss:.4f}")
        print(f"Average Risk      : {avg_risk:.3f}")
        print(f"Average Uncertainty : {avg_uncertainty:.3f}")
        print(f"Epsilon           : {self.epsilon:.3f}")

        print("Action Counts")

        for action_id, action_name in rl1.ACTIONS.items():

            count = stats["actions"].count(action_id)

            print(f"   {action_name:<22}: {count}")

        print("=" * 70)


    def save_best_model(
        self,
        current_reward,
        best_reward,
        filename="./models/best_policy_model.pt"
    ):

        if current_reward > best_reward:

            torch.save(
                self.policy_net.state_dict(),
                filename
            )

            print(
                f"\nNew best model saved "
                f"(Reward={current_reward:.3f})"
            )

            return current_reward

        return best_reward


    def print_buffer_statistics(self):

        print("\nReplay Buffer Statistics")
        print("------------------------")
        print(f"Stored transitions : {len(self.memory)}")
        print(f"Capacity           : {rl1.BUFFER_SIZE}")
        print(f"Fill Percentage    : {(len(self.memory)/rl1.BUFFER_SIZE)*100:.2f}%")