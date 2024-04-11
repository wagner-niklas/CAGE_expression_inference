# Circumplex Affect Guided Emotion Inference 

# Realtime Emotion Inference Supported By The Circumplex Model Utilizing Expression Recognition

### Keywords: User experience, Emotion Inference, FER, Expression Recgonition, Emotion Recognition, Supervised Learning, Computer Vision, Data Set Comparison, Autonomous driving

If you use this repository or any of its contents please cite our Paper: 
CAGE: Circumplex Affect Guided Emotion Inference

### Abstract: 
Understanding emotions is a task of interest across multiple disciplines, especially for improving user experiences. Contrary to the common perception, it has been shown that emotions are not discrete entities but instead exist along a continuum. People understand discrete emotions differently due to a variety of factors, including cultural background, individual experiences and cognitive biases. Therefore, most approaches to emotion understanding, particularly those relying on discrete categories, are inherently biased. In this paper, we present a comparative indepth analysis of two common datasets (AffectNet and EMOTIC) equipped with the components of the circumplex model of affect. Further, we propose a model for prediction of facial expression tailored for lightweight applications. Using a small-scaled MaxViT-based model architecture, we evaluate the impact of discrete emotion category labels in training with the continuous valence and arousal labels. We show that considering valence and arousal in addition to discrete category labels helps to significantly improve emotion prediction. The proposed model outperforms the current state-of-the-art models on AffectNet, establishing it as the best-performing model for inferring valence and arousal achieving a 7% lower RMSE.

### Model inference on a video: 
![](https://github.com/wagner-niklas/KIT_FacialEmotionRecognition/blob/main/Honnold_AffectNet7VA_short.gif)

### Tasks of this project:

[1] Implement live video emotion guessing discrete

[2] Extent code to guess the continuous values of the circumplex model of affect

[3] Test model performance on AffectNet and EMOTIC

[3] Adapt live code to recognize emotions from multiple faces at the same time and the seat they are sitting in (Seat 1, 2 or 3)

[4] Live test emotion recognition on persons watching a video

[5] Research methods for validating and improving results for future work
