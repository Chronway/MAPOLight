实验环境配置
软件/硬件	配置
CPU	Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
GPU	NVIDIA TITAN RTX 24G
内存	64G
操作系统	Ubuntu 20.04
编程语言	Python 3.10.12
仿真平台	SUMO 1.19.0
强化学习平台 Ray 2.6.3
		Pytorch 2.0.1+cu118


环境安装步骤 !! 因为这个版本附近的一些基础库(如numpy)代码改动较大，导致依赖很混乱，一定要按照以下步骤来 !! 即使pip警告部分包的版本不兼容也没事，换成兼容的版本反而会报错：
1. pip install ray[all]==2.6.3
2. pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
3. pip install -r requirements.txt

安装完后的环境可以参考ref_requirements.txt