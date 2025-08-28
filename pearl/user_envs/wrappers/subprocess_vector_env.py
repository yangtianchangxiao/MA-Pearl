#!/usr/bin/env python3
"""
Subprocess vector environment for Pearl - following HARL/Stable-Baselines3 pattern.
Each environment runs in its own CPU process with Pipe communication.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from typing import List, Callable, Any, Tuple

# Fix CUDA multiprocessing compatibility
mp.set_start_method('spawn', force=True)


class CloudpickleWrapper:
    """Uses cloudpickle for multiprocessing compatibility."""
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def pearl_worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker process function - runs environment in separate CPU process.
    This is the standard RL multiprocessing pattern.
    """
    parent_remote.close()
    
    # Create environment in worker process
    env = env_fn_wrapper.x()
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == "step":
                action = data
                result = env.step(action)
                
                # 关键修复：episode结束时自动reset
                if result.terminated or result.truncated:
                    next_obs, _ = env.reset()
                    observation_to_send = next_obs
                else:
                    observation_to_send = result.observation
                
                remote.send({
                    'observation': observation_to_send,
                    'reward': result.reward,
                    'terminated': result.terminated,
                    'truncated': result.truncated,
                    'info': getattr(result, 'info', {})
                })
                
            elif cmd == "reset":
                obs, action_space = env.reset()
                remote.send({
                    'observation': obs,
                    'action_space': action_space
                })
                
            elif cmd == "get_spaces":
                remote.send({
                    'observation_space': env.observation_space,
                    'action_space': getattr(env, 'action_space', None)
                })
                
            elif cmd == "close":
                if hasattr(env, 'close'):
                    env.close()
                remote.close()
                break
                
        except Exception as e:
            print(f"Worker error: {e}")
            remote.send({'error': str(e)})
            break


class SubprocVectorEnv:
    """
    Pearl-compatible subprocess vector environment.
    Follows HARL/Stable-Baselines3 multiprocessing pattern.
    """
    
    def __init__(self, env_fns: List[Callable]):
        """
        Args:
            env_fns: List of functions that create environments
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        
        # Start worker processes
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = Process(
                target=pearl_worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            process.daemon = True  # Die with main process
            process.start()
            self.processes.append(process)
        
        # Close work remotes in main process
        for work_remote in self.work_remotes:
            work_remote.close()
        
        # Get environment spaces from first worker
        self.remotes[0].send(("get_spaces", None))
        spaces = self.remotes[0].recv()
        self.observation_space = spaces['observation_space']
        self.action_space = spaces['action_space']
        
        print(f"✅ SubprocVectorEnv: {self.num_envs} processes started")
    
    def reset(self) -> Tuple[List[np.ndarray], Any]:
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(("reset", None))
        
        results = [remote.recv() for remote in self.remotes]
        observations = [result['observation'] for result in results]
        action_space = results[0]['action_space']
        
        return observations, action_space
    
    def step_async(self, actions: List[np.ndarray]):
        """Send actions to all workers asynchronously."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True
    
    def step_wait(self):
        """Wait for all workers to complete their steps."""
        if not self.waiting:
            raise RuntimeError("step_wait() called without step_async()")
        
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        observations = []
        rewards = []
        terminated = []
        truncated = []
        infos = []
        
        for result in results:
            if 'error' in result:
                raise RuntimeError(f"Worker error: {result['error']}")
            
            observations.append(result['observation'])
            rewards.append(result['reward'])
            terminated.append(result['terminated'])
            truncated.append(result['truncated'])
            infos.append(result['info'])
        
        return observations, rewards, terminated, truncated, infos
    
    def step(self, actions: List[np.ndarray]):
        """Synchronous step - convenience method."""
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        """Close all worker processes."""
        if self.closed:
            return
        
        if self.waiting:
            try:
                for remote in self.remotes:
                    remote.recv()
            except:
                pass
        
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except:
                pass
        
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
        
        for remote in self.remotes:
            remote.close()
        
        self.closed = True
        print(f"✅ SubprocVectorEnv: All {self.num_envs} processes closed")