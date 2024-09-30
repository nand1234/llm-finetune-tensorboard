# llm-finetune-tensorboard
Repo to logg training and validation logs to tensorboard


- create virtual env using python3 -m venv venv
- activate virtual environment source venv/bin/activate
- install dependencies pip3 install requirements.txt
- run python3 main.py to generate the weights
- open new terminal and run to luanch tensorboard ->  tensorboard --logdir=./logs
- navigate to http://localhost:6006/
