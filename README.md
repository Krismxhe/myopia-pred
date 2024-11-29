# myopia-pred

### Introduction
The project aims to classify fundus images into 4 categories - Non-myopia/Low-level myopia/Moderate myopia/Serious myopia.

### Pretrain
```
cd project_repo
mv ./scripts/encoder_pretrain.py ./
python encoder_pretrain.py
```

### Fine-tune
```
python myopia_pred_train.py
```

### Evaluation
```
python myopia_pred_eval.py
```