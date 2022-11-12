# generative-deep-learning
 复现generative-deep-learning中的示例

### 有用的script
```shell
jupyter nbconvert --to script test.ipynb

nohup python test.py &

ps -aux | grep "test.py"

kill -9 PID

zip -r test.zip ./test
```

### 世界模型 WorldModel
- 收集随机 rollout 数据

```shell
python ./world_model/generate_data.py car_racing --total_episodes 2000 --time_steps 300
```
- 训练VAE
```shell
python ./world_model/train_vae.py --new_model
```
- 收集用于训练RNN的数据
```shell
python ./world_model/generate_rnn_data.py
```
- 训练RNN
```shell
python ./world_model/train_rnn.py --new_model
```
- 训练Controller
```shell
# subprocess.py > class Popen > __init__ > shell=True

python ./world_model/train_controller.py car_racing --num_worker 4 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25

python ./world_model/train_controller.py car_racing --num_worker 4 --num_worker_trial 2 --num_episode 4 --max_length 1000 --eval_steps 25 --dream_mode 1
```
- 查看训练结果
```shell
python ./world_model/base/model.py --filename [json filename] --render_mode --record_video
```
