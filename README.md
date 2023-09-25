# rwkv_cuda
## rwkv并行推理，c++实现cuda算子。（初学）
终端输入下列代码，预编译cuda算子。预编译的cuda算子可向下兼容，直接打包可在不同机器上运行。（12.1 可兼容 11.8）特别注意cuda与torch版本需要一致
```
pip install ./cuda
```
编译成功
![image](https://github.com/JL-er/rwkv_cuda/assets/139205286/d2906400-2883-4e78-9edd-93f7c9008cf5)
编译失败遇到下列错误，将setup.py 中的CppExtension 替换成CUDAExtension即可
![image](https://github.com/JL-er/rwkv_cuda/assets/139205286/e08e55aa-f260-41eb-9fa1-5250c8578ee2)

直接使用test.py 进行测试
generate 可以直接实例测试，需要修改你的model路径
