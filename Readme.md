FCN 8, 16, 32를 구현하여 결과를 비교해보았습니다.

- VOC2012 Dataset을 공식 홈페이지에서 다운받고 불러와서 

256 x 256 x 3으로 resize한 뒤 one hot encoding ( 256 x 256 x 21 ) 하였습니다.

![image](https://user-images.githubusercontent.com/46465539/73245505-b3ef1200-41ef-11ea-9799-c88f5444e200.png)


- 본 이미지대로 모델을 구현한 뒤 softmax_cross_entropy loss로 학습시켰습니다.

![image](https://user-images.githubusercontent.com/46465539/73245554-d84aee80-41ef-11ea-9bcc-07a4833fc36d.png)

- 학습 시킨 후 test를 진행하였을 때의 결과입니다. 위쪽 부터 fcn32, 16, 8, GT 이미지 입니다.

![image](https://user-images.githubusercontent.com/46465539/73245599-f1ec3600-41ef-11ea-8e09-6010c0e71011.png)

