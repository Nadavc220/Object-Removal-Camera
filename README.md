# Object-Removal-Camera

This is an academic project. The main idea is to create an algorithm which recieves a static video which is crowded by people
and outputs a single "clean" image with every thing else besides the people using a Detection network (YOLO V3) and classic image processing ideas to recreate the image with "clean" detected paches.
 

### Results Videos
The input videos can be found in the [videos page](videos/).
The output images can be found in the [results page](results/).

### Project Flow Description

If you are interested in learning about the work flow of this project you can read the [project description file](project description.pdf).

### Animal Image Translation

![](results/animal.jpg)

### Street Scene Translation

![](results/street.jpg)

### Yosemite Summer to Winter Translation (HD)

![](results/summer2winter_yosemite.jpg)

### Example-guided Image Translation

![](results/example_guided.jpg)

### Other Implementations

[MUNIT-Tensorflow](https://github.com/taki0112/MUNIT-Tensorflow) by [Junho Kim](https://github.com/taki0112)

[MUNIT-keras](https://github.com/shaoanlu/MUNIT-keras) by [shaoanlu](https://github.com/shaoanlu)

### Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{huang2018munit,
  title={Multimodal Unsupervised Image-to-image Translation},
  author={Huang, Xun and Liu, Ming-Yu and Belongie, Serge and Kautz, Jan},
  booktitle={ECCV},
  year={2018}
}
```
