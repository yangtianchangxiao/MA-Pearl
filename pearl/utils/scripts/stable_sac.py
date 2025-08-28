# Stable SAC with gradient clipping for robotic manipulation
from typing import Any
import torch
import torch.nn.utils as utils
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.replay_buffers.transition import TransitionBatch


class StableContinuousSoftActorCritic(ContinuousSoftActorCritic):
    """
    SAC with gradient clipping and weight decay for stability
    """
    
    def __init__(self, grad_clip_norm: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.grad_clip_norm = grad_clip_norm
        
        print(f"✅ Stable SAC initialized:")
        print(f"   Gradient Clip Norm: {grad_clip_norm}")
    
    def learn(self, replay_buffer) -> dict[str, Any]:
        """Enhanced learning with gradient clipping - Pearl interface"""
        
        # Use parent's learn method but add gradient clipping
        # Sample batch
        batch = replay_buffer.sample(self._batch_size)
        
        # Compute losses
        critic_loss = self._critic_loss(batch)
        actor_loss = self._actor_loss(batch)
        
        # Critic update with gradient clipping
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Apply gradient clipping to critic
        critic_grad_norm = utils.clip_grad_norm_(
            self._critic.parameters(), 
            self.grad_clip_norm
        )
        
        self._critic_optimizer.step()
        
        # Actor update with gradient clipping  
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Apply gradient clipping to actor
        actor_grad_norm = utils.clip_grad_norm_(
            self._actor.parameters(),
            self.grad_clip_norm
        )
        
        self._actor_optimizer.step()
        
        # Update target networks
        self._update_target_networks()
        
        # Handle entropy coefficient update
        entropy_coef_loss = torch.tensor(0.0)
        if self._is_entropy_autotune:
            entropy_coef_loss = self._entropy_coef_loss()
            self._entropy_optimizer.zero_grad()
            entropy_coef_loss.backward()
            
            # Gradient clipping for entropy coefficient
            utils.clip_grad_norm_(
                [self._log_entropy_coef],
                self.grad_clip_norm
            )
            
            self._entropy_optimizer.step()
        
        # Return enhanced metrics
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(), 
            "entropy_coef_loss": entropy_coef_loss.item(),
            "entropy_coef": self._entropy_coef.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_grad_norm": actor_grad_norm.item(),
        }