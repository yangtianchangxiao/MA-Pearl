from setuptools import setup, find_packages

setup(
    name="robot_catch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.26.3",
        "numpy",
        "matplotlib",
        "openrl",
        "torch",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="A robotic arm catching environment for reinforcement learning",
)
