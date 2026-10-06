"""TrainingFactory: run pretrain -> midtrain -> RL in order (the "AI factory")."""
from .pretrain import PretrainRunner
from .midtrain_runner import MidtrainRunner
from .rl_runner import RLRunner


class TrainingFactory:
    def run_all(self, pretrain_steps=50, midtrain_steps=100, rl_prompts=20):
        pre = PretrainRunner().run(n_steps=pretrain_steps)
        mid = MidtrainRunner().run(n_steps=midtrain_steps)
        rl = RLRunner().run(n_prompts=rl_prompts)
        return {"pretrain": pre, "midtrain": mid, "rl": rl}
