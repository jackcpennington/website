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
pd.set_option('max_colwidth', None)
```


```python
df = pd.read_csv("fifa21_male2.csv")
```


```python
df.head()
```




<table border="1">
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
      <th>Club Logo</th>
      <th>Flag Photo</th>
      <th>POT</th>
      <th>Team &amp; Contract</th>
      <th>Height</th>
      <th>Weight</th>
      <th>foot</th>
      <th>Growth</th>
      <th>Joined</th>
      <th>Loan Date End</th>
      <th>Value</th>
      <th>Wage</th>
      <th>Release Clause</th>
      <th>Contract</th>
      <th>Attacking</th>
      <th>Crossing</th>
      <th>Finishing</th>
      <th>Heading Accuracy</th>
      <th>Short Passing</th>
      <th>Volleys</th>
      <th>Skill</th>
      <th>Dribbling</th>
      <th>Curve</th>
      <th>FK Accuracy</th>
      <th>Long Passing</th>
      <th>Ball Control</th>
      <th>Movement</th>
      <th>Acceleration</th>
      <th>Sprint Speed</th>
      <th>Agility</th>
      <th>Reactions</th>
      <th>Balance</th>
      <th>Power</th>
      <th>Shot Power</th>
      <th>Jumping</th>
      <th>Stamina</th>
      <th>Strength</th>
      <th>Long Shots</th>
      <th>Mentality</th>
      <th>Aggression</th>
      <th>Interceptions</th>
      <th>Positioning</th>
      <th>Vision</th>
      <th>Penalties</th>
      <th>Composure</th>
      <th>Defending</th>
      <th>Marking</th>
      <th>Standing Tackle</th>
      <th>Sliding Tackle</th>
      <th>Goalkeeping</th>
      <th>GK Diving</th>
      <th>GK Handling</th>
      <th>GK Kicking</th>
      <th>GK Positioning</th>
      <th>GK Reflexes</th>
      <th>Total Stats</th>
      <th>Base Stats</th>
      <th>W/F</th>
      <th>SM</th>
      <th>A/W</th>
      <th>D/W</th>
      <th>IR</th>
      <th>PAC</th>
      <th>SHO</th>
      <th>PAS</th>
      <th>DRI</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
      <th>LS</th>
      <th>ST</th>
      <th>RS</th>
      <th>LW</th>
      <th>LF</th>
      <th>CF</th>
      <th>RF</th>
      <th>RW</th>
      <th>LAM</th>
      <th>CAM</th>
      <th>RAM</th>
      <th>LM</th>
      <th>LCM</th>
      <th>CM</th>
      <th>RCM</th>
      <th>RM</th>
      <th>LWB</th>
      <th>LDM</th>
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
      <td>https://cdn.sofifa.com/teams/55/light_60.png</td>
      <td>https://cdn.sofifa.com/flags/it.png</td>
      <td>69</td>
      <td>Udinese 2008 ~ 2016</td>
      <td>6'0"</td>
      <td>181lbs</td>
      <td>Left</td>
      <td>0</td>
      <td>Jul 1, 2008</td>
      <td>NaN</td>
      <td>€625K</td>
      <td>€7K</td>
      <td>€0</td>
      <td>2008 ~ 2016</td>
      <td>313</td>
      <td>75</td>
      <td>50</td>
      <td>59</td>
      <td>71</td>
      <td>58.0</td>
      <td>338</td>
      <td>73</td>
      <td>65.0</td>
      <td>60</td>
      <td>69</td>
      <td>71</td>
      <td>347</td>
      <td>68</td>
      <td>74</td>
      <td>68.0</td>
      <td>69</td>
      <td>68.0</td>
      <td>347</td>
      <td>74</td>
      <td>68.0</td>
      <td>69</td>
      <td>68</td>
      <td>68</td>
      <td>320</td>
      <td>72</td>
      <td>69.0</td>
      <td>63.0</td>
      <td>66.0</td>
      <td>50</td>
      <td>NaN</td>
      <td>208</td>
      <td>70</td>
      <td>69</td>
      <td>69.0</td>
      <td>56</td>
      <td>14</td>
      <td>5</td>
      <td>15</td>
      <td>10</td>
      <td>12</td>
      <td>1929</td>
      <td>408</td>
      <td>3 ★</td>
      <td>2★</td>
      <td>Medium</td>
      <td>High</td>
      <td>2 ★</td>
      <td>71</td>
      <td>59</td>
      <td>70</td>
      <td>71</td>
      <td>68</td>
      <td>69</td>
      <td>4</td>
      <td>65+0</td>
      <td>65+0</td>
      <td>65+0</td>
      <td>68+0</td>
      <td>67+0</td>
      <td>67+0</td>
      <td>67+0</td>
      <td>68+0</td>
      <td>68+0</td>
      <td>68+0</td>
      <td>68+0</td>
      <td>69+0</td>
      <td>69+0</td>
      <td>69+0</td>
      <td>69+0</td>
      <td>69+0</td>
      <td>71+-2</td>
      <td>70+-1</td>
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
      <td>https://cdn.sofifa.com/teams/2013/light_60.png</td>
      <td>https://cdn.sofifa.com/flags/es.png</td>
      <td>71</td>
      <td>KAS Eupen 2014 ~ 2019</td>
      <td>5'10"</td>
      <td>143lbs</td>
      <td>Right</td>
      <td>0</td>
      <td>Jul 19, 2014</td>
      <td>NaN</td>
      <td>€600K</td>
      <td>€7K</td>
      <td>€1.1M</td>
      <td>2014 ~ 2019</td>
      <td>337</td>
      <td>68</td>
      <td>64</td>
      <td>61</td>
      <td>76</td>
      <td>68.0</td>
      <td>369</td>
      <td>69</td>
      <td>79.0</td>
      <td>79</td>
      <td>71</td>
      <td>71</td>
      <td>305</td>
      <td>56</td>
      <td>50</td>
      <td>62.0</td>
      <td>65</td>
      <td>72.0</td>
      <td>324</td>
      <td>75</td>
      <td>54.0</td>
      <td>64</td>
      <td>60</td>
      <td>71</td>
      <td>362</td>
      <td>71</td>
      <td>71.0</td>
      <td>72.0</td>
      <td>73.0</td>
      <td>75</td>
      <td>79.0</td>
      <td>153</td>
      <td>70</td>
      <td>43</td>
      <td>40.0</td>
      <td>56</td>
      <td>9</td>
      <td>12</td>
      <td>13</td>
      <td>11</td>
      <td>11</td>
      <td>1906</td>
      <td>385</td>
      <td>4 ★</td>
      <td>3★</td>
      <td>Medium</td>
      <td>Medium</td>
      <td>1 ★</td>
      <td>53</td>
      <td>69</td>
      <td>73</td>
      <td>69</td>
      <td>58</td>
      <td>63</td>
      <td>4</td>
      <td>67+1</td>
      <td>67+1</td>
      <td>67+1</td>
      <td>67+0</td>
      <td>68+0</td>
      <td>68+0</td>
      <td>68+0</td>
      <td>67+0</td>
      <td>70+1</td>
      <td>70+1</td>
      <td>70+1</td>
      <td>68+1</td>
      <td>70+1</td>
      <td>70+1</td>
      <td>70+1</td>
      <td>68+1</td>
      <td>62+1</td>
      <td>66+1</td>
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
      <td>https://cdn.sofifa.com/teams/1800/light_60.png</td>
      <td>https://cdn.sofifa.com/flags/gb-eng.png</td>
      <td>71</td>
      <td>Coventry City 2016 ~ 2020</td>
      <td>5'9"</td>
      <td>161lbs</td>
      <td>Right</td>
      <td>0</td>
      <td>Jan 7, 2016</td>
      <td>NaN</td>
      <td>€1.1M</td>
      <td>€15K</td>
      <td>€0</td>
      <td>2016 ~ 2020</td>
      <td>337</td>
      <td>80</td>
      <td>64</td>
      <td>41</td>
      <td>77</td>
      <td>75.0</td>
      <td>387</td>
      <td>79</td>
      <td>84.0</td>
      <td>77</td>
      <td>69</td>
      <td>78</td>
      <td>295</td>
      <td>48</td>
      <td>42</td>
      <td>71.0</td>
      <td>59</td>
      <td>75.0</td>
      <td>284</td>
      <td>72</td>
      <td>58.0</td>
      <td>29</td>
      <td>56</td>
      <td>69</td>
      <td>317</td>
      <td>69</td>
      <td>39.0</td>
      <td>69.0</td>
      <td>74.0</td>
      <td>66</td>
      <td>NaN</td>
      <td>99</td>
      <td>35</td>
      <td>34</td>
      <td>30.0</td>
      <td>51</td>
      <td>9</td>
      <td>6</td>
      <td>13</td>
      <td>16</td>
      <td>7</td>
      <td>1770</td>
      <td>354</td>
      <td>4 ★</td>
      <td>4★</td>
      <td>Medium</td>
      <td>Low</td>
      <td>2 ★</td>
      <td>45</td>
      <td>68</td>
      <td>76</td>
      <td>77</td>
      <td>36</td>
      <td>52</td>
      <td>11</td>
      <td>64+0</td>
      <td>64+0</td>
      <td>64+0</td>
      <td>70+0</td>
      <td>69+0</td>
      <td>69+0</td>
      <td>69+0</td>
      <td>70+0</td>
      <td>71+0</td>
      <td>71+0</td>
      <td>71+0</td>
      <td>68+0</td>
      <td>66+0</td>
      <td>66+0</td>
      <td>66+0</td>
      <td>68+0</td>
      <td>52+0</td>
      <td>54+0</td>
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
      <td>https://cdn.sofifa.com/teams/106/light_60.png</td>
      <td>https://cdn.sofifa.com/flags/tt.png</td>
      <td>82</td>
      <td>Sunderland 2009</td>
      <td>5'11"</td>
      <td>165lbs</td>
      <td>Right</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>€0</td>
      <td>€0</td>
      <td>€0</td>
      <td>2009</td>
      <td>264</td>
      <td>54</td>
      <td>70</td>
      <td>60</td>
      <td>80</td>
      <td>NaN</td>
      <td>255</td>
      <td>68</td>
      <td>NaN</td>
      <td>46</td>
      <td>64</td>
      <td>77</td>
      <td>176</td>
      <td>59</td>
      <td>62</td>
      <td>NaN</td>
      <td>55</td>
      <td>NaN</td>
      <td>239</td>
      <td>63</td>
      <td>NaN</td>
      <td>51</td>
      <td>66</td>
      <td>59</td>
      <td>271</td>
      <td>59</td>
      <td>70.0</td>
      <td>72.0</td>
      <td>NaN</td>
      <td>70</td>
      <td>NaN</td>
      <td>75</td>
      <td>34</td>
      <td>41</td>
      <td>NaN</td>
      <td>68</td>
      <td>5</td>
      <td>21</td>
      <td>64</td>
      <td>21</td>
      <td>21</td>
      <td>1348</td>
      <td>369</td>
      <td>3 ★</td>
      <td>1★</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1 ★</td>
      <td>61</td>
      <td>66</td>
      <td>66</td>
      <td>69</td>
      <td>47</td>
      <td>60</td>
      <td>3</td>
      <td>67+0</td>
      <td>67+0</td>
      <td>67+0</td>
      <td>66+0</td>
      <td>67+0</td>
      <td>67+0</td>
      <td>67+0</td>
      <td>66+0</td>
      <td>70+0</td>
      <td>70+0</td>
      <td>70+0</td>
      <td>66+0</td>
      <td>68+0</td>
      <td>68+0</td>
      <td>68+0</td>
      <td>66+0</td>
      <td>56+0</td>
      <td>65+0</td>
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
      <td>https://cdn.sofifa.com/teams/101146/light_60.png</td>
      <td>https://cdn.sofifa.com/flags/es.png</td>
      <td>81</td>
      <td>Vissel Kobe 2018 ~ 2021</td>
      <td>5'7"</td>
      <td>150lbs</td>
      <td>Right</td>
      <td>0</td>
      <td>Jul 16, 2018</td>
      <td>NaN</td>
      <td>€5.5M</td>
      <td>€12K</td>
      <td>€7.2M</td>
      <td>2018 ~ 2021</td>
      <td>367</td>
      <td>75</td>
      <td>69</td>
      <td>54</td>
      <td>90</td>
      <td>79.0</td>
      <td>408</td>
      <td>85</td>
      <td>80.0</td>
      <td>70</td>
      <td>83</td>
      <td>90</td>
      <td>346</td>
      <td>61</td>
      <td>56</td>
      <td>79.0</td>
      <td>75</td>
      <td>75.0</td>
      <td>297</td>
      <td>67</td>
      <td>40.0</td>
      <td>58</td>
      <td>62</td>
      <td>70</td>
      <td>370</td>
      <td>58</td>
      <td>70.0</td>
      <td>78.0</td>
      <td>93.0</td>
      <td>71</td>
      <td>89.0</td>
      <td>181</td>
      <td>68</td>
      <td>57</td>
      <td>56.0</td>
      <td>45</td>
      <td>6</td>
      <td>13</td>
      <td>6</td>
      <td>13</td>
      <td>7</td>
      <td>2014</td>
      <td>420</td>
      <td>4 ★</td>
      <td>4★</td>
      <td>High</td>
      <td>Medium</td>
      <td>4 ★</td>
      <td>58</td>
      <td>70</td>
      <td>85</td>
      <td>85</td>
      <td>63</td>
      <td>59</td>
      <td>149</td>
      <td>72+3</td>
      <td>72+3</td>
      <td>72+3</td>
      <td>79+0</td>
      <td>79+0</td>
      <td>79+0</td>
      <td>79+0</td>
      <td>79+0</td>
      <td>82+-1</td>
      <td>82+-1</td>
      <td>82+-1</td>
      <td>79+2</td>
      <td>81+0</td>
      <td>81+0</td>
      <td>81+0</td>
      <td>79+2</td>
      <td>70+3</td>
      <td>73+3</td>
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
