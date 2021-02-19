# "pytorch-widedeep, deep learning for tabular data III: the deeptabular component"
> a flexible package to combine tabular data with text and images using wide and deep models.

- author: Javier Rodriguez
- toc: true 
- badges: true
- comments: true


This is the third of a [series](https://jrzaurin.github.io/infinitoml/) of posts introducing [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep), a flexible package to combine tabular data with text and images (that could also be used for "standard" tabular data alone). 

While writing this post I will assume that the reader is not familiar with the previous two [posts](https://jrzaurin.github.io/infinitoml/). Of course, reading them would help, but in order to understand the content of this post and then being able to use `pytorch-widedeep` on tabular data, is not a requirement. 

As I mentioned earlier, `pytorch-widedeep`'s main goal is to facilitate the combination of images and text with tabular data via wide and deep models. To that aim, [wide and deep models](https://pytorch-widedeep.readthedocs.io/en/latest/model_components.html) can be built with up to four model components: `wide`, `deeptabular`, `deeptext` and `deepimage`, that will take care of the different types of input datasets ("standard" tabular, i.e. numerical and categorical features, text and images). This post focuses only on the so called `deeptabular` component, and the 3 different models available in this library that can be used to build that component. Nonetheless, and for completion, I will briefly describe the remaining components first. 

The `wide` component of a wide and deep model is simply a liner model, and in `pytorch-widedeep` such model can be created via the [`Wide`](https://pytorch-widedeep.readthedocs.io/en/latest/model_components.html#pytorch_widedeep.models.wide.Wide) class. In the case of the `deeptext` component, `pytorch-widedeep` offers one model, available via the [`DeepText`](https://pytorch-widedeep.readthedocs.io/en/latest/model_components.html#pytorch_widedeep.models.deep_text.DeepText) class. `DeepText` builds a simple stack of LSTMs, i.e. a standard DL text classifier or regressor, with flexibility regarding the use of pre-trained word embeddings, of a Fully Connected Head (FC-Head), etc. For the `deepimage` component, `pytorch-widedeep` includes two alternatives: a pre-trained Resnet model or a "standard" stack of CNNs to be trained from scratch. The two are available via the [`DeepImage`](https://pytorch-widedeep.readthedocs.io/en/latest/model_components.html#pytorch_widedeep.models.deep_image.DeepImage) class which, as in the case of `DeepText`, offers some flexibility when building the architecture. 

To clarify the use of the term "*model*" and Wide and Deep "*model component*" (in case there is some confusion), let's have a look to the following code:

```python
wide_model = Wide(...)
text_model = DeepText(...)
image_model = DeepImage(...)

# we use the previous models as the wide and deep model components
wdmodel = WideDeep(wide=wide_model, deeptext=text_model, deepimage=image_model)

...
```

Simply, a wide and deep model has model components that are (of course) models in themselves. Note that any of the four wide and deep model components can be a custom model by the user. In fact, while I recommend using the models available in `pytorch-widedeep` for the `wide` and `deeptabular` model components, it is very likely that users will want to use their own models for the `deeptext` and `deepimage `components. That is perfectly possible as long as the custom models have an attribute called `output_dim` with the size of the last layer of activations, so that `WideDeep` can be constructed. In addition, any of the four components can be used independently in isolation. For example, you might want to use just a `wide` component, which is simply a linear model. To that aim, simply:

```python
wide_model = Wide(...)

# this would not be a wide and deep model but just wide
wdmodel = WideDeep(wide=wide_model)

...
```

If you want to learn more about different model components and the models available in `pytorch-widedeep` please, have a look to the [Examples](https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples) folder in the repo, the [documentation](https://pytorch-widedeep.readthedocs.io/en/latest/model_components.html) or the [companion posts](https://jrzaurin.github.io/infinitoml/). Let's now take a deep dive into the models available for the `deeptabular` component


## 1. The `deeptabular` component

As I was developing the package I realised that perhaps one of the most interesting offerings in `pytorch-widedeep` was related to the models available for the `deeptabular` component. Remember than each component can be used independently in isolation. Building a `WideDeep` model comprised only by a `deeptabular` component would be what is normally referred as DL for tabular data. Of course, such model is not a wide and deep model, is "just" deep.

Currently, `pytorch-widedeep` offers three models that can be used as the `deeptabular` component. In order of complexity, these are:

- `TabMlp`: this is very similar to the [tabular model](https://docs.fast.ai/tutorial.tabular.html) in the fantastic [fastai](https://docs.fast.ai/) library, and consists simply in embeddings representing the categorical features, concatenated with the continuous features, and passed then through a MLP.

- `TabRenset`: This is similar to the previous model but the embeddings are passed through a series of ResNet blocks built with dense layers.

- `TabTransformer`: Details on the TabTransformer can be found in: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf). Again, this is similar to the models before but the embeddings are passed through a series of Transformer (encoder) blocks.

A lot has been (and is being) written about the use of DL for tabular data, and certainly each of these models would deserve a post by themselves (the `TabMlp` is a section in the great [fastai book](https://github.com/fastai/fastbook) and the `TabTransformer` was presented in a scientific publication). Here, I will try to describe them with some detail and illustrate their use within `pytorch-widedeep`. A proper benchmark exercise will be carried out in a not-so-distant future. 

### 1.1 `TabMlp`

The following figure illustrates the `TabMlp` model architecture.


![](pytorch-widedeep%3A%20deep%20learning%20for%20tabular%20data_files/markdown_1_local_image_tag_0.png)

**Fig 1**. The `TabMlp`: this is the simples architecture and is very similar to the tabular model available in the fantastic fastai library. In fact, the implementation of the dense layers of the MLP is mostly identical to that in that library. 

The dashed-border boxes indicate that these components are optional. For example, we could use `TabMlp` without categorical components, or without continuous components, if we wanted. 

Let's have a look and see how this model is used


```python
#hide
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```


```python
#collapse-hide
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

adult = pd.read_csv("data/adult/adult.csv.zip")
adult.columns = [c.replace("-", "_") for c in adult.columns]
adult["income_label"] = (adult["income"].apply(lambda x: ">50K" in x)).astype(int)
adult.drop("income", axis=1, inplace=True)

for c in adult.columns:
    if adult[c].dtype == 'O':
        adult[c] = adult[c].apply(lambda x: "unknown" if x == "?" else x)
        adult[c] = adult[c].str.lower()
adult_train, adult_test = train_test_split(adult, test_size=0.2, stratify=adult.income_label)        
```


```python
adult.head()
```




    
![png](pytorch-widedeep%3A%20deep%20learning%20for%20tabular%20data_files/output_4_0.png)
    




```python
# define the embedding and continuous columns, and target
embed_cols = [
    ('workclass', 6), 
    ('education', 8), 
    ('marital_status', 6), 
    ('occupation',8), 
    ('relationship', 6), 
    ('race', 6)]
cont_cols = ["age", "hours_per_week", "fnlwgt", "educational_num"]
target = adult_train["income_label"].values
```


```python
# prepare deeptabular component
from pytorch_widedeep.preprocessing import TabPreprocessor
tab_preprocessor = TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)

X_tab = tab_preprocessor.fit_transform(adult_train)
```

Let's pause for a second, since the code up until here is going to be common to all models with some minor adaptations for the `TabTransformer`. So far, we have simply define the columns that will be represented by embeddings and the numerical (aka continuous) columns. Once they are defined the dataset is prepared with the `TabPreprocessor`. Internally, the preprocessor label encodes the "embedding columns" and standarises the numerical columns. Note that one could chose not to standarise the numerical columns and then use a `BatchNorm1D` layer when building the model. That is also a valid approach. Alternatively, one could use both, as I will. 

At this stage the data is prepared and we are ready to build the model


```python
from pytorch_widedeep.models import TabMlp, WideDeep

tabmlp = TabMlp(
    mlp_hidden_dims=[200, 100],
    column_idx=tab_preprocessor.column_idx,
    embed_input=tab_preprocessor.embeddings_input, 
    continuous_cols=cont_cols,
    batchnorm_cont=True,
)
```

Let's have a look to the model we just built and how it relates to Fig 1


```python
tabmlp
```




    TabMlp(
      (embed_layers): ModuleDict(
        (emb_layer_education): Embedding(17, 8, padding_idx=0)
        (emb_layer_marital_status): Embedding(8, 6, padding_idx=0)
        (emb_layer_occupation): Embedding(16, 8, padding_idx=0)
        (emb_layer_race): Embedding(6, 6, padding_idx=0)
        (emb_layer_relationship): Embedding(7, 6, padding_idx=0)
        (emb_layer_workclass): Embedding(10, 6, padding_idx=0)
      )
      (embedding_dropout): Dropout(p=0.1, inplace=False)
      (norm): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (tab_mlp): MLP(
        (mlp): Sequential(
          (dense_layer_0): Sequential(
            (0): Dropout(p=0.1, inplace=False)
            (1): Linear(in_features=44, out_features=200, bias=True)
            (2): ReLU(inplace=True)
          )
          (dense_layer_1): Sequential(
            (0): Dropout(p=0.1, inplace=False)
            (1): Linear(in_features=200, out_features=100, bias=True)
            (2): ReLU(inplace=True)
          )
        )
      )
    )



As we can see, we have a series of columns that would be represented as embeddings. The embeddings from all this columns are concatenated, forming a tensor of dim `(bsz, 40)` where `bsz` is batch size. Then, the "*batchnormed*" continuous columns are also concatenated, forming a tensor of dim `(bsz, 44)`, that will be passed to the 2-layer MLP `(200 -> 100)`. 

One important thing to mention, common to all models, is that `pytorch-widedeep` models do not build the last connection, i.e. the connection with the output neuron or neurons depending whether this is a regression, binary or multi-class classification. Such connection is built by the `WideDeep` constructor class. This means that even if we wanted to use a single-component model, the model still needs to be built with the `WideDeep` class.  

This is because the library is, a priori, intended to build `WideDeep` models (and hence its name). Once the model is built it is passed to the `Trainer` (as we will see now). The `Trainer` class is coded to receive a parent model of class `WideDeep` with children that are the model components. This is very convenient for a number of aspects in the library. 

Effectively this simply requires one extra line of code. 


```python
model = WideDeep(deeptabular=tabmlp)
```


```python
model
```




    WideDeep(
      (deeptabular): Sequential(
        (0): TabMlp(
          (embed_layers): ModuleDict(
            (emb_layer_education): Embedding(17, 8, padding_idx=0)
            (emb_layer_marital_status): Embedding(8, 6, padding_idx=0)
            (emb_layer_occupation): Embedding(16, 8, padding_idx=0)
            (emb_layer_race): Embedding(6, 6, padding_idx=0)
            (emb_layer_relationship): Embedding(7, 6, padding_idx=0)
            (emb_layer_workclass): Embedding(10, 6, padding_idx=0)
          )
          (embedding_dropout): Dropout(p=0.1, inplace=False)
          (norm): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (tab_mlp): MLP(
            (mlp): Sequential(
              (dense_layer_0): Sequential(
                (0): Dropout(p=0.1, inplace=False)
                (1): Linear(in_features=44, out_features=200, bias=True)
                (2): ReLU(inplace=True)
              )
              (dense_layer_1): Sequential(
                (0): Dropout(p=0.1, inplace=False)
                (1): Linear(in_features=200, out_features=100, bias=True)
                (2): ReLU(inplace=True)
              )
            )
          )
        )
        (1): Linear(in_features=100, out_features=1, bias=True)
      )
    )



As we can see, our `model` has the final connection now and is a model of class `WideDeep` formed by one single component, `deeptabular`, which is a model of class `TabMlp` formed mainly by the `embed_layers` and an MLP very creatively called `tab_mlp`. 

We are now ready to train it. The code below simply runs with defaults. one could use any `torch` optimizer, learning rate schedulers, etc. Just have a look to the [docs](https://pytorch-widedeep.readthedocs.io/en/latest/trainer.html) or the [Examples](https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples) folder in the repo. 


```python
from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import Accuracy

trainer = Trainer(model, objective="binary", metrics=[(Accuracy)])
trainer.fit(X_tab=X_tab, target=target, n_epochs=5, batch_size=256, val_split=0.2) 
```

    epoch 1: 100%|██████████| 123/123 [00:02<00:00, 54.11it/s, loss=0.404, metrics={'acc': 0.804}] 
    valid: 100%|██████████| 31/31 [00:00<00:00, 119.63it/s, loss=0.388, metrics={'acc': 0.8033}]
    epoch 2: 100%|██████████| 123/123 [00:01<00:00, 69.71it/s, loss=0.362, metrics={'acc': 0.8302}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 109.68it/s, loss=0.374, metrics={'acc': 0.8272}]
    epoch 3: 100%|██████████| 123/123 [00:01<00:00, 64.07it/s, loss=0.358, metrics={'acc': 0.8307}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 118.48it/s, loss=0.368, metrics={'acc': 0.8289}]
    epoch 4: 100%|██████████| 123/123 [00:01<00:00, 65.89it/s, loss=0.354, metrics={'acc': 0.8322}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 120.24it/s, loss=0.365, metrics={'acc': 0.8306}]
    epoch 5: 100%|██████████| 123/123 [00:01<00:00, 67.65it/s, loss=0.351, metrics={'acc': 0.8353}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 114.02it/s, loss=0.361, metrics={'acc': 0.8339}]


Once we understand what `TabMlp` does, `TabResnet` should be pretty straightforward

### 1.2 `TabResnet`

The following figure illustrates the `TabResnet` model architecture.

![](pytorch-widedeep%3A%20deep%20learning%20for%20tabular%20data_files/markdown_16_local_image_tag_0.png)

**Fig 2**. The `TabResnet`: this model is similar to the `TabMlp`, but the embeddings (or the concatenation of embeddings and continuous features, normalised or not) are passed through a series of Resnet blocks built with dense layers. The dashed-border boxes indicate that the component is optional and the dashed lines indicate the different paths or connections present depending on which components we decide to include. 

This is probably the most flexible of the three models discussed in this post in the sense that there are many variants one can define via the parameters. For example, we could chose to concatenate the continuous features, normalized or not via a `BatchNorm1d` layer, with the embeddings and then pass the result of such a concatenation trough the series of Resnet blocks. Alternatively, we might prefer to concatenate the continuous features with the results of passing the embeddings through the Resnet blocks. Another optional component is the MLP before the output neuron(s). If not MLP is present, the output from the Resnet blocks or the results of concatenating that output with the continuous features (normalised or not) will be connected directly to the output neuron(s).

Each of the Resnet block is comprised by the following operations:

![](pytorch-widedeep%3A%20deep%20learning%20for%20tabular%20data_files/markdown_16_local_image_tag_1.png)

Fig 3. "Dense" Resnet Block. `b` is the batch size and `d` the dimension of the embeddings.

Let's build a `TabResnet` model:


```python
from pytorch_widedeep.models import TabResnet

tabresnet = TabResnet(
    column_idx=tab_preprocessor.column_idx,
    embed_input=tab_preprocessor.embeddings_input, 
    continuous_cols=cont_cols,
    batchnorm_cont=True,
    blocks_dims=[200, 100, 100],
    mlp_hidden_dims=[100, 50],
)
model = WideDeep(deeptabular=tabresnet)
model
```




    WideDeep(
      (deeptabular): Sequential(
        (0): TabResnet(
          (embed_layers): ModuleDict(
            (emb_layer_education): Embedding(17, 8, padding_idx=0)
            (emb_layer_marital_status): Embedding(8, 6, padding_idx=0)
            (emb_layer_occupation): Embedding(16, 8, padding_idx=0)
            (emb_layer_race): Embedding(6, 6, padding_idx=0)
            (emb_layer_relationship): Embedding(7, 6, padding_idx=0)
            (emb_layer_workclass): Embedding(10, 6, padding_idx=0)
          )
          (embedding_dropout): Dropout(p=0.1, inplace=False)
          (norm): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (tab_resnet): DenseResnet(
            (dense_resnet): Sequential(
              (lin1): Linear(in_features=44, out_features=200, bias=True)
              (bn1): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (block_0): BasicBlock(
                (lin1): Linear(in_features=200, out_features=100, bias=True)
                (bn1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
                (dp): Dropout(p=0.1, inplace=False)
                (lin2): Linear(in_features=100, out_features=100, bias=True)
                (bn2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (resize): Sequential(
                  (0): Linear(in_features=200, out_features=100, bias=True)
                  (1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (block_1): BasicBlock(
                (lin1): Linear(in_features=100, out_features=100, bias=True)
                (bn1): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (leaky_relu): LeakyReLU(negative_slope=0.01, inplace=True)
                (dp): Dropout(p=0.1, inplace=False)
                (lin2): Linear(in_features=100, out_features=100, bias=True)
                (bn2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (tab_resnet_mlp): MLP(
            (mlp): Sequential(
              (dense_layer_0): Sequential(
                (0): Dropout(p=0.1, inplace=False)
                (1): Linear(in_features=100, out_features=100, bias=True)
                (2): ReLU(inplace=True)
              )
              (dense_layer_1): Sequential(
                (0): Dropout(p=0.1, inplace=False)
                (1): Linear(in_features=100, out_features=50, bias=True)
                (2): ReLU(inplace=True)
              )
            )
          )
        )
        (1): Linear(in_features=50, out_features=1, bias=True)
      )
    )



As we did previously with the `TabMlp`, let's "walk through" the model. As we can see, the object `model` is an instance of a `WideDeep` model formed by a single component, `deeptabular` that is a `TabResnet` model. The `TabResnet` model is formed by a series of embeddings that are concatenated themselves, and then further concatenated with the normalised continuous columns. The resulting tensor of dim `(bsz, 44)` is then passed through a `tab_resnet` component, which is comprised by two so-called "dense" Resnet blocks. The output of the Resnet blocks, of dim `(bsz, 100)` is passed through a 2-layer MLP, named `tab_resnet_mlp` and finally "plugged" into the output neuron. In summary: `Embedding` + `DenseResnet` + `MLP`.

To run it, the code is, as one might expect identical to the one shown before for the `TabMlp`. 


```python
trainer = Trainer(model, objective="binary", metrics=[(Accuracy)])
trainer.fit(X_tab=X_tab, target=target, n_epochs=5, batch_size=256, val_split=0.2) 
```

    epoch 1: 100%|██████████| 123/123 [00:04<00:00, 29.08it/s, loss=0.383, metrics={'acc': 0.8161}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 101.34it/s, loss=0.361, metrics={'acc': 0.8199}]
    epoch 2: 100%|██████████| 123/123 [00:04<00:00, 30.34it/s, loss=0.354, metrics={'acc': 0.8343}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 100.89it/s, loss=0.355, metrics={'acc': 0.8349}]
    epoch 3: 100%|██████████| 123/123 [00:04<00:00, 30.25it/s, loss=0.349, metrics={'acc': 0.8358}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 100.03it/s, loss=0.353, metrics={'acc': 0.8361}]
    epoch 4: 100%|██████████| 123/123 [00:04<00:00, 30.32it/s, loss=0.348, metrics={'acc': 0.8366}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 88.02it/s, loss=0.352, metrics={'acc': 0.8373}]
    epoch 5: 100%|██████████| 123/123 [00:04<00:00, 30.63it/s, loss=0.346, metrics={'acc': 0.8388}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 100.70it/s, loss=0.351, metrics={'acc': 0.8389}]


And now, last but not least, the last addition to the library, the `TabTransformer`.

### 1.3 `TabTransformer`

The `TabTransformer` is described in detail in [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf) [1], by the clever guys at Amazon. Is an entertaining paper that I, of course, strongly recommend if you are going to use this model on your tabular data (and also in general if you are interested in DL for tabular data).

My implementation is not the only one availablle. Given that the model was conceived by the researchers at Amazon, it is also available in their fantastic [`autogluon`](https://github.com/awslabs/autogluon) library (which you should definitely check). In addition, you can find another implementation [here](https://github.com/lucidrains/tab-transformer-pytorch) by Phil Wang, whose entire github is simply outstanding. My implementation is partially inspired by these but has some particularities and adaptations so that it works within the `pytorch-widedeep` package. 

The following figure illustrates the `TabTransformer` model architecture.

![](pytorch-widedeep%3A%20deep%20learning%20for%20tabular%20data_files/markdown_20_local_image_tag_0.png)

**Fig 4**. The `TabTransfomer`, described in [TabTransformer: Tabular Data Modeling Using Contextual Embeddings]. (https://arxiv.org/pdf/2012.06678.pdf). The dashed-border boxes indicate that the component is optional.

As in previous cases, there are a number of variants and details to consider as one builds the model. I will describe some here, but for a full view of all the possible parameters, please, have a look to the [docs](https://pytorch-widedeep.readthedocs.io/en/latest/model_components.html#pytorch_widedeep.models.tab_transformer.TabTransformer). 

I don't want to go into the details of what is a Transformer [2] in this post. There is an overwhelming amount of literature if you wanted to learn about it, with the most popular being perhaps [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html). Also check [this](https://elvissaravia.substack.com/p/learn-about-transformers-a-recipe) post and if you are a math "maniac" you might like this [paper](https://arxiv.org/abs/2007.02876) [3]. However, let me just briefly describe it so I can introduce the little math we will need for this post. In one sentence, a Transformer consists of a multi-head self-attention layer followed by feed-forward layer, with element-wise addition and layer-normalization being done after each layer. 

As most of you will know, a self-attention layer comprises three matrices, Key, Query and Value. Each input categorical column, i.e. embedding, is projected onto these matrices (although see the `fixed_attention` option later in the post) to generate their corresponding key, query and value vectors. Formally, let $K \in R^{e \times d}$, $Q \in R^{e \times d}$ and $V \in R^{e \times d}$ be the Key, Query and Value matrices of the embeddings where $e$ is the embeddings dimension and $d$ is the dimension of all the Key, Query and Value matrices. Then every input categorical column, i.e embedding, attends to all other categorical columns through an attention head: 

$$
Attention(K, Q, V ) = A \cdot V, \hspace{5cm}(1)
$$

where 

$$
A = softmax( \frac{QK^T}{\sqrt{d}} ), \hspace{6cm}(2)
$$

And that is all the math we need. 

As I was thinking in a figure to illustrate a transformer block, I realised that there is a chance that the reader has seen every possible representation/figure. Therefore, I decided to illustrate the transformer block in a way that relates directly to the way it is implemented.

![](pytorch-widedeep%3A%20deep%20learning%20for%20tabular%20data_files/markdown_20_local_image_tag_1.png)

**Fig 5**. The Transfomer block. The letters in parenthesis indicate the dimension of the corresponding tensor after the operation indicated in the corresponding box. For example, the tensor `attn_weights` have dim `(b, h, s, s)`.

As the figure shows, the input tensor ($X$) is projected onto its key, query and value matrices. These are then "*re-arranged into*" the multi-head self-attention layer where each head will attend to part of the embeddings. We then compute $A$ (Eq 2), which is then multiplied by $V$ to obtain what I refer as `attn_score` (Eq 1). `attn_score` is then re-arranged, so that we "*collect*" the attention scores from all the heads, and projected again to obtain the results (`attn_out`), that will be added to the input and normalised (`Y`). Finally `Y` goes through the Feed-Forward layer and a further Add + Norm.

Before moving to the code related to building the model itself, there are a couple of details in the implementation that are worth mentioning. Here we go:

- `FullEmbeddingDropout`: when building a `TabTransformer` model, there is the possibility of dropping entirely the embedding corresponding to a categorical column. This is set by the parameter `full_embed_dropout: bool`, which points to the class [`FullEmbeddingDropout`](https://github.com/jrzaurin/pytorch-widedeep/blob/be96b57f115e4a10fde9bb82c35380a3ac523f52/pytorch_widedeep/models/tab_transformer.py#L153). 


 - `SharedEmbeddings`: when building a `TabTransformer` model, it is possible for all the embeddings that represent a categorical column to share a fraction of their embeddings, or define a common separated embedding per column that will be added to the column's embeddings. 
 
     The idea behind this so-called "*column embedding*" is to enable the model to distinguish the classes in one column from those in the other columns. In other words, we want the model to learn representations not only of the different categorical values in the column, but also of the column itself. This is attained by the `shared_embed` group of parameters: `share_embed : bool`, `add_shared_embed: bool` and `frac_shared_embed: int`. The first simply indicates if embeddings will be shared, the second sets the sharing strategy and the third one the fraction of the embeddings that will be shared, depending on the strategy. They all relate to the class [`SharedEmbeddings`](https://github.com/jrzaurin/pytorch-widedeep/blob/be96b57f115e4a10fde9bb82c35380a3ac523f52/pytorch_widedeep/models/tab_transformer.py#L165)
 
     For example, let's say that we have a categorical column with 5 different categories that will be encoded as embeddings of dim 8. This will result in a lookup table for that column of dim `(5, 8)`. The two sharing strategies are illustrated in Fig 6. 
 
     ![](pytorch-widedeep%3A%20deep%20learning%20for%20tabular%20data_files/markdown_20_local_image_tag_2.png)

    **Fig 6**. The two sharing embeddings strategies. Upper panel: the "*column embedding*" replaces `embedding dim / frac_shared_embed` (4 in this case) of the total embeddings that represent the different values of the categorical column. Lower panel: the "*column embedding*" is added (well, technically broadcasted and added) to the original embedding lookup table. Note that `n_cat` here refers to the number of different categories for this particular column. 

- `fixed_attention`: this in inspired by the [implementation](https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/tab_transformer/modified_transformer.py) at the Autogluon library. When using "fixed attention", the key and query matrices are not the result of any projection of the input tensor $X$, but learnable matrices (referred as `fixed_key` and `fixed_query`) defined separately, as you instantiate the model. `fixed_attention` does not affect how the Value matrix is computed. 

    Let me go through an example with numbers to clarify things. Let's assume we have a dataset with 5 categorical columns that will be encoded by embeddings of dim 4 and we use a batch size (`bsz`) of 6. Figure 7 shows how the key matrix will be computed for a given batch (same applies to the query matrix) with and without fixed attention. 
    
     ![](pytorch-widedeep%3A%20deep%20learning%20for%20tabular%20data_files/markdown_20_local_image_tag_3.png)
    
    **Fig 7**. Key matrix computation for a given batch with and without fixed attention (same applies to the query matrix). The different color tones in the matrices are my attempt to illustrate that, while without fixed attention the key matrix can have different values anywhere in the matrix, with fixed attention the key matrix is the result of the repetition of the "fixed-key" `bsz` times. The input projected layer is, of course, broadcasted aong the `bsz` dimension in the upper panel.

    As I mentioned, this implementation is inspired by that at the Autogluon library. Since the guys at Amazon are the ones that came up with the `TabTransformer`, is only logical to think that they might have found a purpose for this implementation of attention, at least in some cases. However, at the time of writing such purpose is not 100% clear to me. It is known that, in problems like machine translation, most attention heads learn redundant patterns (see e.g. [Alessandro Raganato et al., 2020](https://arxiv.org/abs/2002.10260) [4] and references therein). Therefore, maybe the fixed attention mechanism discussed here helps reducing redundancy for problems involving tabular data.
    
    Overall, the way I interpret it, in layman's terms, is the following: when using fixed attention, the Key and the Query matrices are defined as the model is instantiated, and do not know of the input until the attention weights (`attn_weights`) are multiplied by the value matrix to obtain what I refer as `attn_score` in figure 5. Those attention weights, which are in essence the result of a matrix multiplication between the key and the query matrices (plus softmax and normalization), are going to be the same for all the heads, for all samples in a given batch. Therefore, my interpretation is that when using fixed attention, we reduce the attention capabilities of the transformer, which will focus on less aspects of the inputs, reducing potential redundancies.
        
Anyway, enough speculation. Time to have a look to the code. Note that, since we are going to stack the embeddings (instead of concatenating them) they all must have the same dimensions. Such dimension is set as we build the model instead that at the pre-processing stage. To avoid input format conflicts we use the `for_tabtransformer` parameter at pre-processing time.    


```python
embed_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race']
tab_preprocessor = TabPreprocessor(
    embed_cols=embed_cols, 
    continuous_cols=cont_cols, 
    for_tabtransformer=True)

X_tab = tab_preprocessor.fit_transform(adult_train)
```

    /Users/javier/.pyenv/versions/3.7.9/envs/wdposts/lib/python3.7/site-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)



```python
from pytorch_widedeep.models import TabTransformer

tabtransformer = TabTransformer(
    column_idx=tab_preprocessor.column_idx,
    embed_input=tab_preprocessor.embeddings_input, 
    continuous_cols=cont_cols,
    shared_embed=True,
    num_blocks=3,
)
model = WideDeep(deeptabular=tabtransformer)
model
```




    WideDeep(
      (deeptabular): Sequential(
        (0): TabTransformer(
          (embed_layers): ModuleDict(
            (emb_layer_education): SharedEmbeddings(
              (embed): Embedding(17, 32, padding_idx=0)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (emb_layer_marital_status): SharedEmbeddings(
              (embed): Embedding(8, 32, padding_idx=0)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (emb_layer_occupation): SharedEmbeddings(
              (embed): Embedding(16, 32, padding_idx=0)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (emb_layer_race): SharedEmbeddings(
              (embed): Embedding(6, 32, padding_idx=0)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (emb_layer_relationship): SharedEmbeddings(
              (embed): Embedding(7, 32, padding_idx=0)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (emb_layer_workclass): SharedEmbeddings(
              (embed): Embedding(10, 32, padding_idx=0)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (blks): Sequential(
            (block0): TransformerEncoder(
              (self_attn): MultiHeadedAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (inp_proj): Linear(in_features=32, out_features=96, bias=True)
                (out_proj): Linear(in_features=32, out_features=32, bias=True)
              )
              (feed_forward): PositionwiseFF(
                (w_1): Linear(in_features=32, out_features=128, bias=True)
                (w_2): Linear(in_features=128, out_features=32, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (activation): GELU()
              )
              (attn_addnorm): AddNorm(
                (dropout): Dropout(p=0.1, inplace=False)
                (ln): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
              )
              (ff_addnorm): AddNorm(
                (dropout): Dropout(p=0.1, inplace=False)
                (ln): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
              )
            )
            (block1): TransformerEncoder(
              (self_attn): MultiHeadedAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (inp_proj): Linear(in_features=32, out_features=96, bias=True)
                (out_proj): Linear(in_features=32, out_features=32, bias=True)
              )
              (feed_forward): PositionwiseFF(
                (w_1): Linear(in_features=32, out_features=128, bias=True)
                (w_2): Linear(in_features=128, out_features=32, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (activation): GELU()
              )
              (attn_addnorm): AddNorm(
                (dropout): Dropout(p=0.1, inplace=False)
                (ln): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
              )
              (ff_addnorm): AddNorm(
                (dropout): Dropout(p=0.1, inplace=False)
                (ln): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
              )
            )
            (block2): TransformerEncoder(
              (self_attn): MultiHeadedAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (inp_proj): Linear(in_features=32, out_features=96, bias=True)
                (out_proj): Linear(in_features=32, out_features=32, bias=True)
              )
              (feed_forward): PositionwiseFF(
                (w_1): Linear(in_features=32, out_features=128, bias=True)
                (w_2): Linear(in_features=128, out_features=32, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (activation): GELU()
              )
              (attn_addnorm): AddNorm(
                (dropout): Dropout(p=0.1, inplace=False)
                (ln): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
              )
              (ff_addnorm): AddNorm(
                (dropout): Dropout(p=0.1, inplace=False)
                (ln): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (tab_transformer_mlp): MLP(
            (mlp): Sequential(
              (dense_layer_0): Sequential(
                (0): Linear(in_features=196, out_features=784, bias=True)
                (1): ReLU(inplace=True)
                (2): Dropout(p=0.1, inplace=False)
              )
              (dense_layer_1): Sequential(
                (0): Linear(in_features=784, out_features=392, bias=True)
                (1): ReLU(inplace=True)
                (2): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (1): Linear(in_features=392, out_features=1, bias=True)
      )
    )



Let's walk through the model. Initially, and as always, we have the embeddings that will represent the categorical columns. The only particular aspect in this model is that the embeddings are of class `SharedEmbeddings`, which I described before. These embeddings are stacked and passed through three transformer blocks. The output for all the categorical columns is concatenated, resulting in a tensor of dim `(bsz, 192)` where 192 is equal to the number of categorical columns (6) times the embedding dim (32). This tensor is then concatenated with the "layernormed" continuous columns, resulting in a tensor of dim `(bsz, 196)`. As usual, this tensor goes through an MLP and "off we go".

To run it, the code is, as one might expect identical to the one shown before for the `TabMlp` and `TabRenset`.


```python
trainer = Trainer(model, objective="binary", metrics=[(Accuracy)])
trainer.fit(X_tab=X_tab, target=target, n_epochs=5, batch_size=256, val_split=0.2) 
```

    epoch 1: 100%|██████████| 123/123 [00:09<00:00, 12.84it/s, loss=0.371, metrics={'acc': 0.8264}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 37.22it/s, loss=0.355, metrics={'acc': 0.8286}]
    epoch 2: 100%|██████████| 123/123 [00:09<00:00, 12.90it/s, loss=0.35, metrics={'acc': 0.8359}] 
    valid: 100%|██████████| 31/31 [00:00<00:00, 36.94it/s, loss=0.358, metrics={'acc': 0.8348}]
    epoch 3: 100%|██████████| 123/123 [00:09<00:00, 13.01it/s, loss=0.347, metrics={'acc': 0.837}] 
    valid: 100%|██████████| 31/31 [00:00<00:00, 36.82it/s, loss=0.361, metrics={'acc': 0.8342}]
    epoch 4: 100%|██████████| 123/123 [00:09<00:00, 12.88it/s, loss=0.344, metrics={'acc': 0.84}]  
    valid: 100%|██████████| 31/31 [00:00<00:00, 36.70it/s, loss=0.363, metrics={'acc': 0.837}] 
    epoch 5: 100%|██████████| 123/123 [00:09<00:00, 12.63it/s, loss=0.341, metrics={'acc': 0.8412}]
    valid: 100%|██████████| 31/31 [00:00<00:00, 36.74it/s, loss=0.358, metrics={'acc': 0.8387}]


## 2. Conclusion and future work

In this post my intention was to illustrate how one can use `pytorch-widedeep` as a library for "standard DL for tabular data", i.e. without building wide and deep models and for problems that do not involve text and/or images (if you wanted to learn more about the library please visit the [repo](https://github.com/jrzaurin/pytorch-widedeep), the [documentation](https://pytorch-widedeep.readthedocs.io/en/latest/index.html), or the [previous posts](https://jrzaurin.github.io/infinitoml/)).  To that aim the only component that we need is the `deeptabular` component, for which `pytorch-widedeep` comes with 3 models implemented "out of the box": `TabMlp`, `TabResnet` and `TabTransformer`. In this post I have explained their architecture in detail and how to use them within the library. In the no-so-distant future I intend to implement [TabNet](https://arxiv.org/abs/1908.07442) [5], as well as performing a proper benchmarking exercise so I can set robust defaults and then release version `1.0`. Of course, you can help me by using the package in your datasets 🙂. If you found this post useful and you like the library, please give a star to the [repo](https://github.com/jrzaurin/pytorch-widedeep). Other than that, happy coding. 

## 3. References

[1] TabTransformer: Tabular Data Modeling Using Contextual Embeddings. Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin, 2020. [arXiv:2012.06678v1](https://arxiv.org/abs/2012.06678)
 
[2] Attention Is All You Need, Ashish Vaswani, Noam Shazeer, Niki Parmar, et al., 2017. [arXiv:1706.03762v5](https://arxiv.org/abs/1706.03762)

[3] A Mathematical Theory of Attention, James Vuckovic, Aristide Baratin, Remi Tachet des Combes, 2020. [arXiv:2007.02876v2](https://arxiv.org/abs/2007.02876)

[4] Fixed Encoder Self-Attention Patterns in Transformer-Based Machine Translation. Alessandro Raganato, Yves Scherrer, Jörg Tiedemann, 2020. [arXiv:2002.10260v3](https://arxiv.org/abs/2002.10260)

[5] TabNet: Attentive Interpretable Tabular Learning, Sercan O. Arik, Tomas Pfister, [arXiv:1908.07442v5](https://arxiv.org/abs/1908.07442)
