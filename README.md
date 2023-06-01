Neural Style Transfer
==

**Pytorch Implementation Code of "[Image Style Transfer Using Convolutional Neural Networks CVPR2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)"**


| Content | Style | Result |
| --- | --- | --- |
| <img src='./imgs/golden_gate.jpg' width='256'> | <img src='./imgs/starry_night.jpg' width='256'> | <img src='./imgs/ex004.gif' width='256'> |

## Usage
### Requirements
* pytorch: 2.0.1

```bash
python main.py --noise_ratio 0.3 --iteration 1000
```

### Resule per `noise_ratio`

| `0.0` | `0.4` | `0.8` |
| --- | --- | --- |
| <img src='./imgs/ex001.gif' width='256'> | <img src='./imgs/ex003.gif' width='256'> | <img src='./imgs/ex005.gif' width='256'> |
