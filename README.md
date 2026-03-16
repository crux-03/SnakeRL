# SnakeRL
A small project testing reinforcement learning with the classic snake game

## Running
First initialize a venv, then install requirements with:
```
pip install -r requirements.txt
```
Due to me optimizing the snake simulation by writing it in rust, you first need to build the game first:
```
cd snake_rs
maturin develop --release
```
After this you can run either `agent.py` to train or `watch.py` to watch it play.

This repository ships with three different types of models. Test them with these individual commands:
```
python watch.py --type mlp
python watch.py --type conv
python watch.py --type hybrid
```