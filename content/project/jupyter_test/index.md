---
title: Display Jupyter Notebooks with Academic
subtitle: Learn how to blog in Academic using Jupyter notebooks
summary: Learn how to blog in Academic using Jupyter notebooks
authors:
- admin
tags: []
categories: []
date: "2019-02-05T00:00:00Z"
lastMod: "2019-09-05T00:00:00Z"
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ""
  focal_point: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---


```python
import pandas as pd

```


```python
df = pd.read_csv("fifa21_male2.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>OVA</th>
      <th>Nationality</th>
      <th>Club</th>
      <th>BOV</th>
      <th>BP</th>
      <th>Position</th>
      <th>Player Photo</th>
      <th>...</th>
      <th>CDM</th>
      <th>RDM</th>
      <th>RWB</th>
      <th>LB</th>
      <th>LCB</th>
      <th>CB</th>
      <th>RCB</th>
      <th>RB</th>
      <th>GK</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>G. Pasquale</td>
      <td>33</td>
      <td>69</td>
      <td>Italy</td>
      <td>Udinese</td>
      <td>71</td>
      <td>LWB</td>
      <td>LM</td>
      <td>https://cdn.sofifa.com/players/000/002/16_120.png</td>
      <td>...</td>
      <td>70+-1</td>
      <td>70+-1</td>
      <td>71+-2</td>
      <td>70+-1</td>
      <td>69+0</td>
      <td>69+0</td>
      <td>69+0</td>
      <td>70+-1</td>
      <td>17+0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>Luis García</td>
      <td>37</td>
      <td>71</td>
      <td>Spain</td>
      <td>KAS Eupen</td>
      <td>70</td>
      <td>CM</td>
      <td>CM CAM CDM</td>
      <td>https://cdn.sofifa.com/players/000/016/19_120.png</td>
      <td>...</td>
      <td>66+1</td>
      <td>66+1</td>
      <td>62+1</td>
      <td>60+1</td>
      <td>60+1</td>
      <td>60+1</td>
      <td>60+1</td>
      <td>60+1</td>
      <td>17+1</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27</td>
      <td>J. Cole</td>
      <td>33</td>
      <td>71</td>
      <td>England</td>
      <td>Coventry City</td>
      <td>71</td>
      <td>CAM</td>
      <td>CAM RM RW LM</td>
      <td>https://cdn.sofifa.com/players/000/027/16_120.png</td>
      <td>...</td>
      <td>54+0</td>
      <td>54+0</td>
      <td>52+0</td>
      <td>47+0</td>
      <td>46+0</td>
      <td>46+0</td>
      <td>46+0</td>
      <td>47+0</td>
      <td>15+0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>D. Yorke</td>
      <td>36</td>
      <td>68</td>
      <td>Trinidad &amp;amp; Tobago</td>
      <td>Sunderland</td>
      <td>70</td>
      <td>ST</td>
      <td>NaN</td>
      <td>https://cdn.sofifa.com/players/000/036/09_120.png</td>
      <td>...</td>
      <td>65+0</td>
      <td>65+0</td>
      <td>56+0</td>
      <td>57+0</td>
      <td>51+0</td>
      <td>51+0</td>
      <td>51+0</td>
      <td>57+0</td>
      <td>22+0</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>Iniesta</td>
      <td>36</td>
      <td>81</td>
      <td>Spain</td>
      <td>Vissel Kobe</td>
      <td>82</td>
      <td>CAM</td>
      <td>CM CAM</td>
      <td>https://cdn.sofifa.com/players/000/041/20_120.png</td>
      <td>...</td>
      <td>73+3</td>
      <td>73+3</td>
      <td>70+3</td>
      <td>67+3</td>
      <td>64+3</td>
      <td>64+3</td>
      <td>64+3</td>
      <td>67+3</td>
      <td>17+3</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 107 columns</p>
</div>




```python

```
