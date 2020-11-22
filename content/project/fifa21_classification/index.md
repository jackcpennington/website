---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Classifying Players Positions From Their FIFA 21 Statistics"
summary: ""
authors: ["Jack Pennington"]
tags: ["Machine Learning", "Classification"]
categories: []
date: 2020-11-22T16:07:35Z

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---
The new FIFA 21 game came out recently, which my flat mate got on release. While playing I thought it would be cool to apply some things I've learnt during Uni, and then I found a dataset of all the players. So I decided to do this project. In the project I will be following a workflow from one of my core textbooks http://index-of.es/Varios-2/Hands%20on%20Machine%20Learning%20with%20Scikit%20Learn%20and%20Tensorflow.pdf

## Framing the Problem

* The **Objective** of this project is to classify players positions based on there statistics and ratings.
* This will be an offline supervised problem
* The performanced will be **measured** by using suitable metrics to calcualte the overall accuracy (e.g accuracy, precision, recall, f1score)
* The minimum performace needed to reach the objective is and accuracy of 3.7% (as there are 27 categories, choosing randomly would have this accuracy)

## Getting the Data
The data I will be using is the Fifa 21 Complete Player Dataset (https://www.kaggle.com/ekrembayar/fifa-21-complete-player-dataset/notebooks). This contains all the info of each player on the FIFA 21 game

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import re

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
```
```python
df = pd.read_csv('fifa21_male2.csv')
df.head()
```