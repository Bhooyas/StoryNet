# A Decoder Based Transformer for Story Generation

![Transformer](https://socialify.git.ci/Bhooyas/StoryNet/image?font=KoHo&language=1&name=1&owner=1&pattern=Circuit%20Board&stargazers=1&theme=Auto)

A simple decoder based transformer trained on some samples `TinyStories Dataset` for generating stories.

## Inference Using the Transformer

The first step would be to clone the project using the following command: -
```
git clone https://github.com/Bhooyas/StoryNet.git
```

The next step is to install the requirements for the project. We do that using the following command: -
```
cd StoryNet
pip install -r requirements.txt
```

Then we can infer from the model using the following script: -
```
python inference.py
```

## Training the Transformer

The first step would be to clone the project using the following command: -
```
git clone https://github.com/Bhooyas/StoryNet.git
```

The next step is to install the requirements for the project. We do that using the following command: -
```
cd StoryNet
pip install -r requirements.txt
```

The configuration of the models can be found in `config.py`.

The next step would be to get the stories and train the tokenizer on it. We do that using the following command: -
```
python process_data.py
```

The next step is to pretrain the transformer: -
```
python pre_train.py
```

The step after this is to train/fine-tune the pretrained transformer: -
```
python train.py
```

Then we can infer the just trained transformer using the following script: -
```
python inference.py
```

**Note**: - The quality of stories generated is quite dependent on the number of stories you train on. The current model was trained on 1,00,000 stories on a `NVIDIA GeForce GTX 1650Ti` which for pretraining took approximately 25 hours.
