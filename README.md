# Circumplex Affect Guided Expression Inference (CAGE)

## Realtime Expression Inference Supported By The Circumplex Model 

### Keywords: User experience, Expression Inference, FER, Expression Recgonition, Emotion Recognition, Supervised Learning, Computer Vision, Data Set Comparison, Autonomous driving
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cage-circumplex-affect-guided-expression/arousal-estimation-on-affectnet)](https://paperswithcode.com/sota/arousal-estimation-on-affectnet?p=cage-circumplex-affect-guided-expression)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cage-circumplex-affect-guided-expression/valence-estimation-on-affectnet)](https://paperswithcode.com/sota/valence-estimation-on-affectnet?p=cage-circumplex-affect-guided-expression)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cage-circumplex-affect-guided-expression/dominance-estimation-on-emotic)](https://paperswithcode.com/sota/dominance-estimation-on-emotic?p=cage-circumplex-affect-guided-expression)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cage-circumplex-affect-guided-expression/arousal-estimation-on-emotic)](https://paperswithcode.com/sota/arousal-estimation-on-emotic?p=cage-circumplex-affect-guided-expression)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cage-circumplex-affect-guided-expression/valence-estimation-on-emotic)](https://paperswithcode.com/sota/valence-estimation-on-emotic?p=cage-circumplex-affect-guided-expression)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cage-circumplex-affect-guided-expression/emotion-recognition-on-emotic)](https://paperswithcode.com/sota/emotion-recognition-on-emotic?p=cage-circumplex-affect-guided-expression)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cage-circumplex-affect-guided-expression/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=cage-circumplex-affect-guided-expression)

### Citation
If you use this repository or any of its contents please consider citiing our Paper: 
[CAGE: Circumplex Affect Guided Expression Inference](https://arxiv.org/abs/2404.14975) 
```
@misc{wagner2024cage,
      title={CAGE: Circumplex Affect Guided Expression Inference}, 
      author={Niklas Wagner and Felix Mätzler and Samed R. Vossberg and Helen Schneider and Svetlana Pavlitska and J. Marius Zöllner},
      year={2024},
      eprint={2404.14975},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Abstract: 
Understanding expressions and emotions is a task of interest across multiple disciplines, especially for improving user experiences. Contrary to the common perception, it has been shown that expressions are not discrete entities but instead exist along a continuum. People understand discrete expressions differently due to a variety of factors, including cultural background, individual experiences and cognitive biases. Therefore, most approaches to expression understanding, particularly those relying on discrete categories, are inherently biased. In this paper, we present a comparative indepth analysis of two common datasets (AffectNet and EMOTIC) equipped with the components of the circumplex model of affect. Further, we propose a model for prediction of facial expression tailored for lightweight applications. Using a small-scaled MaxViT-based model architecture, we evaluate the impact of discrete expression category labels in training with the continuous valence and arousal labels. We show that considering valence and arousal in addition to discrete category labels helps to significantly improve expression prediction. The proposed model outperforms the current state-of-the-art models on AffectNet, establishing it as the best-performing model for inferring valence and arousal achieving a 7% lower RMSE.

### Model inference on a video: 
![](https://github.com/wagner-niklas/KIT_FacialEmotionRecognition/blob/main/Honnold_inference.gif)


### Usage:
To run the version with our best performing model simply cd into the project directory and run: 
Install requirements: 
```
pip install -r requirements.txt
```

Webcam Expression Inference
```
python3 AffectNet_7VA_webcam_inference.py 
```

If you want to train / alter the models you can run one of the python scripts in the directory. 
To run the train scripts, make sure you have the datasets of EMOTIC[[1]](#1) and AffectNet[[2]](#2) downloaded and saved in the right directory.
The Datasets are not publically available and access has to be requested ([EMOTIC, 2019](https://s3.sunai.uoc.edu/emotic/download.html)) ([AffectNet, 2017](http://mohammadmahoor.com/affectnet/))



<a id="1">[1]</a> 
R. Kosti, J.M. Álvarez, A. Recasens and A. Lapedriza, "Context based emotion recognition using emotic dataset", IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2019.

<a id="2">[2]</a> 
Ali Mollahosseini, Behzad Hasani and Mohammad H. Mahoor, "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild," in IEEE Transactions on Affective Computing, vol. 10, no. 1, pp. 18-31, 1 Jan.-March 2019, doi: 10.1109/TAFFC.2017.2740923.'

### Tasks of this project:

[1] Implement live video expression inference discrete

[2] Extend code to guess the continuous values of the circumplex model of affect

[3] Test model performance on AffectNet and EMOTIC

[4] Live test expression inference

[5] Research methods for validating and improving results for future work
