#  以強化學習實現避障功能-以unity作為環境


以unity建立模擬環境，從螢幕中擷取影像並輸入到神經網路中，神經網路輸出轉按鍵直接控制unity。

---
## unity環境

[envs.exe](https://github.com/zhu913104/gogodick/tree/master/unity%20environment)是建置過的unity模擬環境

使用w,a,s,d鍵來操縱方向、r鍵來重製環境。

---
## python環境

1. numpy==1.13.1
2. tensorflow==1.4.0

---
## 使用說明

1. 打開[envs.exe](https://github.com/zhu913104/gogodick/tree/master/unity%20environment)會出現模擬畫面(立體影像)
2. 選擇[A2C.py](https://github.com/zhu913104/gogodick/blob/master/A2C.py)，[DQN.py](https://github.com/zhu913104/gogodick/blob/master/DQN.py)，[PG.py](https://github.com/zhu913104/gogodick/blob/master/PG.py)
任一個來執行神經網路。
3. [saved_networks](https://github.com/zhu913104/gogodick/tree/master/saved_networks)中有預先訓練好的網路，log中紀錄了loss、reword等等訓練紀錄。

---
## 參見

[MorvanZhou  Reinforcement-learning](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)

[Sentdex  pygta5](https://github.com/sentdex/pygta5)
