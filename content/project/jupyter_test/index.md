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
pd.set_option('max_columns', None)
```


```python
df = pd.read_csv("fifa21_male2.csv")
```


```python
df.head().to_markdown()
```




    '|    |   ID | Name        |   Age |   OVA | Nationality           | Club          |   BOV | BP   | Position     | Player Photo                                      | Club Logo                                        | Flag Photo                              |   POT | Team & Contract           | Height   | Weight   | foot   |   Growth | Joined       |   Loan Date End | Value   | Wage   | Release Clause   | Contract    |   Attacking |   Crossing |   Finishing |   Heading Accuracy |   Short Passing |   Volleys |   Skill |   Dribbling |   Curve |   FK Accuracy |   Long Passing |   Ball Control |   Movement |   Acceleration |   Sprint Speed |   Agility |   Reactions |   Balance |   Power |   Shot Power |   Jumping |   Stamina |   Strength |   Long Shots |   Mentality |   Aggression |   Interceptions |   Positioning |   Vision |   Penalties |   Composure |   Defending |   Marking |   Standing Tackle |   Sliding Tackle |   Goalkeeping |   GK Diving |   GK Handling |   GK Kicking |   GK Positioning |   GK Reflexes |   Total Stats |   Base Stats | W/F   | SM   | A/W    | D/W    | IR   |   PAC |   SHO |   PAS |   DRI |   DEF |   PHY |   Hits | LS   | ST   | RS   | LW   | LF   | CF   | RF   | RW   | LAM   | CAM   | RAM   | LM   | LCM   | CM   | RCM   | RM   | LWB   | LDM   | CDM   | RDM   | RWB   | LB    | LCB   | CB   | RCB   | RB    | GK   | Gender   |\n|---:|-----:|:------------|------:|------:|:----------------------|:--------------|------:|:-----|:-------------|:--------------------------------------------------|:-------------------------------------------------|:----------------------------------------|------:|:--------------------------|:---------|:---------|:-------|---------:|:-------------|----------------:|:--------|:-------|:-----------------|:------------|------------:|-----------:|------------:|-------------------:|----------------:|----------:|--------:|------------:|--------:|--------------:|---------------:|---------------:|-----------:|---------------:|---------------:|----------:|------------:|----------:|--------:|-------------:|----------:|----------:|-----------:|-------------:|------------:|-------------:|----------------:|--------------:|---------:|------------:|------------:|------------:|----------:|------------------:|-----------------:|--------------:|------------:|--------------:|-------------:|-----------------:|--------------:|--------------:|-------------:|:------|:-----|:-------|:-------|:-----|------:|------:|------:|------:|------:|------:|-------:|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:------|:------|:------|:-----|:------|:-----|:------|:-----|:------|:------|:------|:------|:------|:------|:------|:-----|:------|:------|:-----|:---------|\n|  0 |    2 | G. Pasquale |    33 |    69 | Italy                 | Udinese       |    71 | LWB  | LM           | https://cdn.sofifa.com/players/000/002/16_120.png | https://cdn.sofifa.com/teams/55/light_60.png     | https://cdn.sofifa.com/flags/it.png     |    69 | Udinese 2008 ~ 2016       | 6\'0"     | 181lbs   | Left   |        0 | Jul 1, 2008  |             nan | €625K   | €7K    | €0               | 2008 ~ 2016 |         313 |         75 |          50 |                 59 |              71 |        58 |     338 |          73 |      65 |            60 |             69 |             71 |        347 |             68 |             74 |        68 |          69 |        68 |     347 |           74 |        68 |        69 |         68 |           68 |         320 |           72 |              69 |            63 |       66 |          50 |         nan |         208 |        70 |                69 |               69 |            56 |          14 |             5 |           15 |               10 |            12 |          1929 |          408 | 3 ★   | 2★   | Medium | High   | 2 ★  |    71 |    59 |    70 |    71 |    68 |    69 |      4 | 65+0 | 65+0 | 65+0 | 68+0 | 67+0 | 67+0 | 67+0 | 68+0 | 68+0  | 68+0  | 68+0  | 69+0 | 69+0  | 69+0 | 69+0  | 69+0 | 71+-2 | 70+-1 | 70+-1 | 70+-1 | 71+-2 | 70+-1 | 69+0  | 69+0 | 69+0  | 70+-1 | 17+0 | Male     |\n|  1 |   16 | Luis García |    37 |    71 | Spain                 | KAS Eupen     |    70 | CM   | CM CAM CDM   | https://cdn.sofifa.com/players/000/016/19_120.png | https://cdn.sofifa.com/teams/2013/light_60.png   | https://cdn.sofifa.com/flags/es.png     |    71 | KAS Eupen 2014 ~ 2019     | 5\'10"    | 143lbs   | Right  |        0 | Jul 19, 2014 |             nan | €600K   | €7K    | €1.1M            | 2014 ~ 2019 |         337 |         68 |          64 |                 61 |              76 |        68 |     369 |          69 |      79 |            79 |             71 |             71 |        305 |             56 |             50 |        62 |          65 |        72 |     324 |           75 |        54 |        64 |         60 |           71 |         362 |           71 |              71 |            72 |       73 |          75 |          79 |         153 |        70 |                43 |               40 |            56 |           9 |            12 |           13 |               11 |            11 |          1906 |          385 | 4 ★   | 3★   | Medium | Medium | 1 ★  |    53 |    69 |    73 |    69 |    58 |    63 |      4 | 67+1 | 67+1 | 67+1 | 67+0 | 68+0 | 68+0 | 68+0 | 67+0 | 70+1  | 70+1  | 70+1  | 68+1 | 70+1  | 70+1 | 70+1  | 68+1 | 62+1  | 66+1  | 66+1  | 66+1  | 62+1  | 60+1  | 60+1  | 60+1 | 60+1  | 60+1  | 17+1 | Male     |\n|  2 |   27 | J. Cole     |    33 |    71 | England               | Coventry City |    71 | CAM  | CAM RM RW LM | https://cdn.sofifa.com/players/000/027/16_120.png | https://cdn.sofifa.com/teams/1800/light_60.png   | https://cdn.sofifa.com/flags/gb-eng.png |    71 | Coventry City 2016 ~ 2020 | 5\'9"     | 161lbs   | Right  |        0 | Jan 7, 2016  |             nan | €1.1M   | €15K   | €0               | 2016 ~ 2020 |         337 |         80 |          64 |                 41 |              77 |        75 |     387 |          79 |      84 |            77 |             69 |             78 |        295 |             48 |             42 |        71 |          59 |        75 |     284 |           72 |        58 |        29 |         56 |           69 |         317 |           69 |              39 |            69 |       74 |          66 |         nan |          99 |        35 |                34 |               30 |            51 |           9 |             6 |           13 |               16 |             7 |          1770 |          354 | 4 ★   | 4★   | Medium | Low    | 2 ★  |    45 |    68 |    76 |    77 |    36 |    52 |     11 | 64+0 | 64+0 | 64+0 | 70+0 | 69+0 | 69+0 | 69+0 | 70+0 | 71+0  | 71+0  | 71+0  | 68+0 | 66+0  | 66+0 | 66+0  | 68+0 | 52+0  | 54+0  | 54+0  | 54+0  | 52+0  | 47+0  | 46+0  | 46+0 | 46+0  | 47+0  | 15+0 | Male     |\n|  3 |   36 | D. Yorke    |    36 |    68 | Trinidad &amp; Tobago | Sunderland    |    70 | ST   | nan          | https://cdn.sofifa.com/players/000/036/09_120.png | https://cdn.sofifa.com/teams/106/light_60.png    | https://cdn.sofifa.com/flags/tt.png     |    82 | Sunderland 2009           | 5\'11"    | 165lbs   | Right  |       14 | nan          |             nan | €0      | €0     | €0               | 2009        |         264 |         54 |          70 |                 60 |              80 |       nan |     255 |          68 |     nan |            46 |             64 |             77 |        176 |             59 |             62 |       nan |          55 |       nan |     239 |           63 |       nan |        51 |         66 |           59 |         271 |           59 |              70 |            72 |      nan |          70 |         nan |          75 |        34 |                41 |              nan |            68 |           5 |            21 |           64 |               21 |            21 |          1348 |          369 | 3 ★   | 1★   | nan    | nan    | 1 ★  |    61 |    66 |    66 |    69 |    47 |    60 |      3 | 67+0 | 67+0 | 67+0 | 66+0 | 67+0 | 67+0 | 67+0 | 66+0 | 70+0  | 70+0  | 70+0  | 66+0 | 68+0  | 68+0 | 68+0  | 66+0 | 56+0  | 65+0  | 65+0  | 65+0  | 56+0  | 57+0  | 51+0  | 51+0 | 51+0  | 57+0  | 22+0 | Male     |\n|  4 |   41 | Iniesta     |    36 |    81 | Spain                 | Vissel Kobe   |    82 | CAM  | CM CAM       | https://cdn.sofifa.com/players/000/041/20_120.png | https://cdn.sofifa.com/teams/101146/light_60.png | https://cdn.sofifa.com/flags/es.png     |    81 | Vissel Kobe 2018 ~ 2021   | 5\'7"     | 150lbs   | Right  |        0 | Jul 16, 2018 |             nan | €5.5M   | €12K   | €7.2M            | 2018 ~ 2021 |         367 |         75 |          69 |                 54 |              90 |        79 |     408 |          85 |      80 |            70 |             83 |             90 |        346 |             61 |             56 |        79 |          75 |        75 |     297 |           67 |        40 |        58 |         62 |           70 |         370 |           58 |              70 |            78 |       93 |          71 |          89 |         181 |        68 |                57 |               56 |            45 |           6 |            13 |            6 |               13 |             7 |          2014 |          420 | 4 ★   | 4★   | High   | Medium | 4 ★  |    58 |    70 |    85 |    85 |    63 |    59 |    149 | 72+3 | 72+3 | 72+3 | 79+0 | 79+0 | 79+0 | 79+0 | 79+0 | 82+-1 | 82+-1 | 82+-1 | 79+2 | 81+0  | 81+0 | 81+0  | 79+2 | 70+3  | 73+3  | 73+3  | 73+3  | 70+3  | 67+3  | 64+3  | 64+3 | 64+3  | 67+3  | 17+3 | Male     |'




```python

```
