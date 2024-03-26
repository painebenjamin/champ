
<h1 align='Center'>Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance</h1>

<div align='Center'>
    <a href='https://github.com/ShenhaoZhu' target='_blank'>Shenhao Zhu</a><sup>*1</sup>&emsp;
    <a href='https://github.com/Leoooo333' target='_blank'>Junming Leo Chen</a><sup>*2</sup>&emsp;
    <a href='https://github.com/daizuozhuo' target='_blank'>Zuozhuo Dai</a><sup>3</sup>&emsp;
    <a href='https://ai3.fudan.edu.cn/info/1088/1266.htm' target='_blank'>Yinghui Xu</a><sup>2</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>1</sup>&emsp;
    <a href='https://yoyo000.github.io/' target='_blank'>Yao Yao</a><sup>1</sup>&emsp;
    <a href='http://zhuhao.cc/home/' target='_blank'>Hao Zhu</a><sup>+1</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home' target='_blank'>Siyu Zhu</a><sup>+2</sup>
</div>
<div align='Center'>
    <sup>1</sup>Nanjing University <sup>2</sup>Fudan University <sup>3</sup>Alibaba Group
</div>
<div align='Center'>
    <sup>*</sup>Equal Contribution
    <sup>+</sup>Corresponding Author
</div>

<div align='Center'>
    <a href='https://fudan-generative-vision.github.io/champ/#/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2403.14781'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://youtu.be/2XVsy9tQRAY'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
</div>

https://github.com/fudan-generative-vision/champ/assets/82803297/b4571be6-dfb0-4926-8440-3db229ebd4aa

# Framework
![framework](assets/framework.jpg)

# Installation
- System requirement: Ubuntu20.04
- Tested GPUs: A100, RTX3090

Create conda environment: 
```bash
conda create -n champ python=3.10
conda activate champ
```
Install packages with `pip`:
```bash
pip install git+https://github.com/painebenjamin/champ.git
```

# Inference
To inference, simply instantiate the pipeline and pass your arguments.

```py
from champ import CHAMPPipeline

pipeline = CHAMPPipeline.from_pretrained(
  "benjamin-paine/champ",
  torch_dtype=torch.float16,
  variant="fp16",
  device="cuda"
).to("cuda", dtype=torch.float16)

result = pipeline(
  reference: PIL.Image.Image,
  guidance: Dict[str, List[PIL.Image.Image]],
  width: int,
  height: int,
  video_length: int,
  num_inference_steps: int,
  guidance_scale: float
).videos
# Result is a list of PIL Images
```

# Example
One small set of example data is provided in this repository, with a script to execute. Here is the command for inference:
```bash
python inference.py
```
Animation results will be saved as `output.mp4`.

# Acknowledgements
We thank the authors of [MagicAnimate](https://github.com/magic-research/magic-animate), [Animate Anyone](https://github.com/HumanAIGC/AnimateAnyone), and [AnimateDiff](https://github.com/guoyww/AnimateDiff) for their excellent work. Our project is built upon [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), and we are grateful for their open-source contributions.

# Citation
If you find our work useful for your research, please consider citing the paper:
```
@misc{zhu2024champ,
      title={Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance}, 
      author={Shenhao Zhu and Junming Leo Chen and Zuozhuo Dai and Yinghui Xu and Xun Cao and Yao Yao and Hao Zhu and Siyu Zhu},
      year={2024},
      eprint={2403.14781},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Opportunities available
Multiple research positions are open at the **Generative Vision Lab, Fudan University**! Include:
* Research assistant
* Postdoctoral researcher
* PhD candidate
* Master students

Interested individuals are encouraged to contact us at [siyuzhu@fudan.edu.cn](mailto://siyuzhu@fudan.edu.cn) for further information.
