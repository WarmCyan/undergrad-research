tmux new-session -d -s research

tmux new-window -t research:10 -n notebook
tmux send-keys -t research:notebook jupyter Space notebook Space './notebook' Enter 

tmux new-window -t research:9 -n tensorboard
tmux send-keys -t research:tensorboard 'tensorboard --logdir=tensorboard_data' Enter

tmux send-keys -t research:0 './startup/appstart.sh' Enter
tmux send-keys -t research:0 './startup/shellstart.sh' Enter

tmux a -t research:0
