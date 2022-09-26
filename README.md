# Im2Oil: Stroke-Based Oil Painting Rendering with Linearly Controllable Fineness Via Adaptive Sampling
This is the official implementation of the **ACM MM 2022: ACM International Conference on Multimedia** accepted paper "**Im2Oil: Stroke-Based Oil Painting Rendering with Linearly Controllable Fineness Via Adaptive Sampling**".   
Our paper's official version is available at: https://ojs.aaai.org/index.php/AAAI/article/view/16140 
Our Supplementary Material (PDF) is available at Baidu Netdisk (百度网盘) https://pan.baidu.com/s/1ZZEmpuPXRNBPvG0WHbbPKA. The extraction code (提取码) is `1234`.  
## Authors
- Zhengyan Tong (Shanghai Jiao Tong University Master): 此论文idea提供者、代码原作者、论文主笔者，主要从事计算机视觉方面的研究。发表此论文时为上海交通大学电子信息与电气工程学院硕士研究生。  
Email: 418004@sjtu.edu.cn
- Xiaohang Wang (Shanghai Jiao Tong University Master)
- Shengchao Yuan (Shanghai Jiao Tong University Master)
- Xuanhong Chen (Shanghai Jiao Tong University Ph. D.)  
- JunJie Wang (Shanghai Jiao Tong University Master)
- Xiangzhong Fang (Shanghai Jiao Tong University Professor)
## Acknowledgments
- I am extremely grateful to my coworkers' advice, comments, and encouragement, which greatly improves this work. 
## Examples
We give three examples that can be run directly (the hyperparameters of these three examples have been fixed).  
### Quick start
- To draw the cat: `python cat.py`
- To draw the dog: `python dog.py`
- To draw the girl: `python girl.py`
### Results
| | | |
| --- | --- | --- |
| <video src="https://user-images.githubusercontent.com/47803475/192312722-7841b5e6-b836-4b23-b63d-56f9b907a9f8.mp4" controls="controls"></video> | <video src="https://github.com/TZYSJTU/Im2Oil/blob/main/videos/A9.mp4" controls="controls"></video> | <video src="https://github.com/TZYSJTU/Im2Oil/blob/main/videos/B8.mp4" controls="controls"></video> |
| <video src="https://github.com/TZYSJTU/Im2Oil/blob/main/videos/L8.mp4" controls="controls"></video> | <video src="https://github.com/TZYSJTU/Im2Oil/blob/main/videos/P1.mp4" controls="controls"></video> | <video src="https://github.com/TZYSJTU/Im2Oil/blob/main/videos/P4.mp4" controls="controls"></video> |
| <video src="https://github.com/TZYSJTU/Im2Oil/blob/main/videos/S1.mp4" controls="controls"></video> | <video src="https://github.com/TZYSJTU/Im2Oil/blob/main/videos/S6.mp4" controls="controls"></video> | <video src="https://github.com/TZYSJTU/Im2Oil/blob/main/videos/S9.mp4" controls="controls"></video> |


## Instructions
To draw arbitrary input: `python process_order.py`. Of course you need to adjust the following parameters.
### Hyperparameters
- `input_path = './input/your file'` Input image path
- `output_path = './output'` Do not change this
- `n =  10` Gray-scale quantization order
- `period = 5` Line(stroke) width
- `direction =  10` Direction quantization order
- `Freq = 100` Save the drawing process every `Freq` strokes are drawn
- `deepen =  1` Edge map's intensity. The bigger, the darker.
- `transTone = False` Do not change this
- `kernel_radius = 3` Edge tangent flow kernel size, do not change this
- `iter_time = 15` Edge tangent flow kernel iterations times, do not change this
- `background_dir = None`  Whether fix the drawing direction in the background, this value could be `None` or an integer between `(0~180)`
- `CLAHE = True` Whether input uses CLAHE (Do not change this)
- `edge_CLAHE = True` Whether edge map uses CLAHE (Do not change this)
- `draw_new = True` Do not change this
- `random_order = False` Use random drawing order if `True`
- `ETF_order = True` Use the drawing order described in our paper if `True`
- `process_visible = True` Whether show the drawing process 

In our supplementary material (PDF), we explain these hyperparameters in more detail and we show more comparisons with existing pencil drawing algorithms. We also offer more
results of our method. Our Supplementary Material is available at Baidu Netdisk (百度网盘) https://pan.baidu.com/s/1ZZEmpuPXRNBPvG0WHbbPKA. The extraction code (提取码) is `1234`.

# To cite our paper
```
@inproceedings{tong2021sketch,
  title={Sketch generation with drawing process guided by vector flow and grayscale},
  author={Tong, Zhengyan and Chen, Xuanhong and Ni, Bingbing and Wang, Xiaohang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={1},
  pages={609--616},
  year={2021}
}
```

### Videos



https://user-images.githubusercontent.com/47803475/192312722-7841b5e6-b836-4b23-b63d-56f9b907a9f8.mp4

https://user-images.githubusercontent.com/47803475/192313223-1094902c-a565-43bb-9d35-156c709f6ff8.mp4

https://user-images.githubusercontent.com/47803475/192313388-49aa915a-3e18-4010-9af3-380a78f9a65f.mp4

https://user-images.githubusercontent.com/47803475/192313447-0190f4fd-9a8e-40d1-9a6b-91d0c368abd9.mp4

https://user-images.githubusercontent.com/47803475/192313506-c436fb69-f902-4a13-8965-dd6f68334281.mp4

https://user-images.githubusercontent.com/47803475/192313540-55c79205-025e-456e-9f21-7ac33a23ccf3.mp4

https://user-images.githubusercontent.com/47803475/192313571-7def236a-b2f2-4749-a8a6-564fd0c21f00.mp4

https://user-images.githubusercontent.com/47803475/192313593-27c6075b-1a9c-42b9-8743-58cebd3cbbf2.mp4

https://user-images.githubusercontent.com/47803475/192313613-e6355f5d-1ffb-4ae2-8fee-9e6e8125e1e7.mp4


