# Data Mining Project - Survival Analytics

Project Details: Advanced lung cancer survival analysis
1. Determine survival chance with Kaplar Meyer Estimation.
2. Diference between Male and Female population

Important links: https://pypi.org/project/lifelines --> Documentation to Survival Analysis

Dataset details:
inst: Institution code
time: Survival time in days
status: censoring status 1=censored, 2=dead
age: Age in years
sex: Male=1 Female=2
ph.ecog: ECOG performance score as rated by the physician. 0=asymptomatic, 1= symptomatic but completely ambulatory, 2= in bed <50% of the day, 3= in bed > 50%of the day but not bedbound, 4 = bedbound
ph.karno: Karnofsky performance score (bad=0-good=100) rated by physician
pat.karno: Karnofsky performance score as rated by patient
meal.cal: Calories consumed at meals
wt.loss: Weight loss in last six months

Dependent Variable = Status


```python
# Import libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter # Kaplan-Meier model
from lifelines.statistics import logrank_test  # Statistical test
```


```python
# Import Dataset from statsmodel api
dataset = sm.datasets.get_rdataset("lung", package="survival").data
```


```python
# Transforming dependent variables based on inputs from the document
dataset.loc[dataset.status == 1, "status"] = 0
dataset.loc[dataset.status == 2, "status"] = 1
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
