# Data Wrangling Final
Chris Perry

Read in the data:

``` python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

batting = pd.read_csv("/Users/chrisperry/Downloads/Batting.csv")
people = pd.read_csv("/Users/chrisperry/Downloads/People.csv")
salary = pd.read_csv("/Users/chrisperry/Downloads/Salaries.csv")
```

merge data:

``` python
merge1 = pd.merge(batting, people, how='left', left_on='playerID', right_on='playerID')
merge2 = pd.merge(merge1, salary, how='left', left_on='playerID', right_on='playerID')
merge2.columns
merge2.head
```

    <bound method NDFrame.head of          playerID  yearID_x  stint teamID_x lgID_x    G  G_batting   AB   R  \
    0       aardsda01      2004      1      SFN     NL   11        NaN    0   0   
    1       aardsda01      2004      1      SFN     NL   11        NaN    0   0   
    2       aardsda01      2004      1      SFN     NL   11        NaN    0   0   
    3       aardsda01      2004      1      SFN     NL   11        NaN    0   0   
    4       aardsda01      2004      1      SFN     NL   11        NaN    0   0   
    ...           ...       ...    ...      ...    ...  ...        ...  ...  ..   
    394325  zwilldu01      1915      1      CHF     FL  150        NaN  548  65   
    394326  zwilldu01      1916      1      CHN     NL   35        NaN   53   4   
    394327   zychto01      2015      1      SEA     AL   13        NaN    0   0   
    394328   zychto01      2016      1      SEA     AL   12        NaN    0   0   
    394329   zychto01      2017      1      SEA     AL   45        NaN    0   0   

              H  ...  bats  throws       debut    bbrefID   finalGame   retroID  \
    0         0  ...     R       R  2004-04-06  aardsda01  2015-08-23  aardd001   
    1         0  ...     R       R  2004-04-06  aardsda01  2015-08-23  aardd001   
    2         0  ...     R       R  2004-04-06  aardsda01  2015-08-23  aardd001   
    3         0  ...     R       R  2004-04-06  aardsda01  2015-08-23  aardd001   
    4         0  ...     R       R  2004-04-06  aardsda01  2015-08-23  aardd001   
    ...     ...  ...   ...     ...         ...        ...         ...       ...   
    394325  157  ...     L       L  1910-08-14  zwilldu01  1916-07-12  zwild101   
    394326    6  ...     L       L  1910-08-14  zwilldu01  1916-07-12  zwild101   
    394327    0  ...     R       R  2015-09-04   zychto01  2017-08-19  zycht001   
    394328    0  ...     R       R  2015-09-04   zychto01  2017-08-19  zycht001   
    394329    0  ...     R       R  2015-09-04   zychto01  2017-08-19  zycht001   

            yearID_y  teamID_y  lgID_y     salary  
    0         2004.0       SFN      NL   300000.0  
    1         2007.0       CHA      AL   387500.0  
    2         2008.0       BOS      AL   403250.0  
    3         2009.0       SEA      AL   419000.0  
    4         2010.0       SEA      AL  2750000.0  
    ...          ...       ...     ...        ...  
    394325       NaN       NaN     NaN        NaN  
    394326       NaN       NaN     NaN        NaN  
    394327    2016.0       SEA      AL   511000.0  
    394328    2016.0       SEA      AL   511000.0  
    394329    2016.0       SEA      AL   511000.0  

    [394330 rows x 52 columns]>

Filter merged data by last 5 years to view up-to-date stats:

``` python
five = merge2[merge2['yearID_x'] >= merge2['yearID_x'].max() - 4]
five.head
```

    <bound method NDFrame.head of          playerID  yearID_x  stint teamID_x lgID_x    G  G_batting   AB   R  \
    198      abadfe01      2021      1      BAL     AL   16        NaN    0   0   
    199      abadfe01      2021      1      BAL     AL   16        NaN    0   0   
    200      abadfe01      2021      1      BAL     AL   16        NaN    0   0   
    201      abadfe01      2021      1      BAL     AL   16        NaN    0   0   
    202      abadfe01      2021      1      BAL     AL   16        NaN    0   0   
    ...           ...       ...    ...      ...    ...  ...        ...  ...  ..   
    394281  zuninmi01      2021      1      TBA     AL  109        NaN  333  64   
    394282  zuninmi01      2022      1      TBA     AL   36        NaN  115   7   
    394283  zuninmi01      2022      1      TBA     AL   36        NaN  115   7   
    394284  zuninmi01      2023      1      CLE     AL   42       42.0  124  11   
    394285  zuninmi01      2023      1      CLE     AL   42       42.0  124  11   

             H  ...  bats  throws       debut    bbrefID   finalGame   retroID  \
    198      0  ...     L       L  2010-07-28   abadfe01  2021-10-01  abadf001   
    199      0  ...     L       L  2010-07-28   abadfe01  2021-10-01  abadf001   
    200      0  ...     L       L  2010-07-28   abadfe01  2021-10-01  abadf001   
    201      0  ...     L       L  2010-07-28   abadfe01  2021-10-01  abadf001   
    202      0  ...     L       L  2010-07-28   abadfe01  2021-10-01  abadf001   
    ...     ..  ...   ...     ...         ...        ...         ...       ...   
    394281  72  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   
    394282  17  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   
    394283  17  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   
    394284  22  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   
    394285  22  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   

            yearID_y  teamID_y  lgID_y     salary  
    198       2011.0       HOU      NL   418000.0  
    199       2012.0       HOU      NL   485000.0  
    200       2014.0       OAK      AL   525900.0  
    201       2015.0       OAK      AL  1087500.0  
    202       2016.0       MIN      AL  1250000.0  
    ...          ...       ...     ...        ...  
    394281    2015.0       SEA      AL   523500.0  
    394282    2014.0       SEA      AL   504100.0  
    394283    2015.0       SEA      AL   523500.0  
    394284    2014.0       SEA      AL   504100.0  
    394285    2015.0       SEA      AL   523500.0  

    [12168 rows x 52 columns]>

Filter by minimum at bats to provide accurate results

``` python
f_ab = five[five['AB'] >= 100].copy()
f_ab.head
```

    <bound method NDFrame.head of          playerID  yearID_x  stint teamID_x lgID_x    G  G_batting   AB   R  \
    650     abramcj01      2022      1      SDN     NL   46        NaN  125  16   
    651     abramcj01      2022      2      WAS     NL   44        NaN  159  17   
    652     abramcj01      2023      1      WAS     NL  151      151.0  563  83   
    653     abramcj01      2024      1      WAS     NL  138      138.0  541  79   
    1007    abreujo02      2020      1      CHA     AL   60        NaN  240  43   
    ...           ...       ...    ...      ...    ...  ...        ...  ...  ..   
    394281  zuninmi01      2021      1      TBA     AL  109        NaN  333  64   
    394282  zuninmi01      2022      1      TBA     AL   36        NaN  115   7   
    394283  zuninmi01      2022      1      TBA     AL   36        NaN  115   7   
    394284  zuninmi01      2023      1      CLE     AL   42       42.0  124  11   
    394285  zuninmi01      2023      1      CLE     AL   42       42.0  124  11   

              H  ...  bats  throws       debut    bbrefID   finalGame   retroID  \
    650      29  ...     L       R  2022-04-08  abramcj01         NaN  abrac001   
    651      41  ...     L       R  2022-04-08  abramcj01         NaN  abrac001   
    652     138  ...     L       R  2022-04-08  abramcj01         NaN  abrac001   
    653     133  ...     L       R  2022-04-08  abramcj01         NaN  abrac001   
    1007     76  ...     R       R  2014-03-31  abreujo02         NaN  abrej003   
    ...     ...  ...   ...     ...         ...        ...         ...       ...   
    394281   72  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   
    394282   17  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   
    394283   17  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   
    394284   22  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   
    394285   22  ...     R       R  2013-06-12  zuninmi01  2023-06-14  zunim001   

            yearID_y  teamID_y  lgID_y     salary  
    650          NaN       NaN     NaN        NaN  
    651          NaN       NaN     NaN        NaN  
    652          NaN       NaN     NaN        NaN  
    653          NaN       NaN     NaN        NaN  
    1007      2014.0       CHA      AL  7000000.0  
    ...          ...       ...     ...        ...  
    394281    2015.0       SEA      AL   523500.0  
    394282    2014.0       SEA      AL   504100.0  
    394283    2015.0       SEA      AL   523500.0  
    394284    2014.0       SEA      AL   504100.0  
    394285    2015.0       SEA      AL   523500.0  

    [3607 rows x 52 columns]>

Create an advanced statistic: OBP - on base percentage

``` python
f_ab['OBP'] = (
    (f_ab['H'] + f_ab['BB'] + f_ab['HBP']) / 
    (f_ab['AB'] + f_ab['BB'] + f_ab['HBP'] + f_ab['SF'])
)

f_ab.columns
```

    Index(['playerID', 'yearID_x', 'stint', 'teamID_x', 'lgID_x', 'G', 'G_batting',
           'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB',
           'HBP', 'SH', 'SF', 'GIDP', 'G_old', 'ID', 'birthYear', 'birthMonth',
           'birthDay', 'birthCity', 'birthCountry', 'birthState', 'deathYear',
           'deathMonth', 'deathDay', 'deathCountry', 'deathState', 'deathCity',
           'nameFirst', 'nameLast', 'nameGiven', 'weight', 'height', 'bats',
           'throws', 'debut', 'bbrefID', 'finalGame', 'retroID', 'yearID_y',
           'teamID_y', 'lgID_y', 'salary', 'OBP'],
          dtype='object')

Visualize OBP and Salary as a scatterplot. Is there a visual
relationship?

``` python
sns.relplot(data = f_ab, x = 'salary', y ='OBP')
plt.show()
```

![](README_files/figure-commonmark/cell-7-output-1.png)

There appears to be little to no correlation between salary and obp

How many players have a salary over 3650000 and an OBP over 0.35?

``` python
over365 = f_ab[(f_ab['salary']> 3650000) & (f_ab['OBP'] > 0.35)]

over365
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | playerID | yearID_x | stint | teamID_x | lgID_x | G | G_batting | AB | R | H | ... | throws | debut | bbrefID | finalGame | retroID | yearID_y | teamID_y | lgID_y | salary | OBP |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 1007 | abreujo02 | 2020 | 1 | CHA | AL | 60 | NaN | 240 | 43 | 76 | ... | R | 2014-03-31 | abreujo02 | NaN | abrej003 | 2014.0 | CHA | AL | 7000000.0 | 0.370229 |
| 1008 | abreujo02 | 2020 | 1 | CHA | AL | 60 | NaN | 240 | 43 | 76 | ... | R | 2014-03-31 | abreujo02 | NaN | abrej003 | 2015.0 | CHA | AL | 8666000.0 | 0.370229 |
| 1009 | abreujo02 | 2020 | 1 | CHA | AL | 60 | NaN | 240 | 43 | 76 | ... | R | 2014-03-31 | abreujo02 | NaN | abrej003 | 2016.0 | CHA | AL | 11666667.0 | 0.370229 |
| 1010 | abreujo02 | 2021 | 1 | CHA | AL | 152 | NaN | 566 | 86 | 148 | ... | R | 2014-03-31 | abreujo02 | NaN | abrej003 | 2014.0 | CHA | AL | 7000000.0 | 0.350531 |
| 1011 | abreujo02 | 2021 | 1 | CHA | AL | 152 | NaN | 566 | 86 | 148 | ... | R | 2014-03-31 | abreujo02 | NaN | abrej003 | 2015.0 | CHA | AL | 8666000.0 | 0.350531 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 367831 | vottojo01 | 2021 | 1 | CIN | NL | 129 | NaN | 448 | 73 | 119 | ... | R | 2007-09-04 | vottojo01 | 2023-10-01 | vottj001 | 2012.0 | CIN | NL | 11410655.0 | 0.375235 |
| 367832 | vottojo01 | 2021 | 1 | CIN | NL | 129 | NaN | 448 | 73 | 119 | ... | R | 2007-09-04 | vottojo01 | 2023-10-01 | vottj001 | 2013.0 | CIN | NL | 18910655.0 | 0.375235 |
| 367833 | vottojo01 | 2021 | 1 | CIN | NL | 129 | NaN | 448 | 73 | 119 | ... | R | 2007-09-04 | vottojo01 | 2023-10-01 | vottj001 | 2014.0 | CIN | NL | 12000000.0 | 0.375235 |
| 367834 | vottojo01 | 2021 | 1 | CIN | NL | 129 | NaN | 448 | 73 | 119 | ... | R | 2007-09-04 | vottojo01 | 2023-10-01 | vottj001 | 2015.0 | CIN | NL | 14000000.0 | 0.375235 |
| 367835 | vottojo01 | 2021 | 1 | CIN | NL | 129 | NaN | 448 | 73 | 119 | ... | R | 2007-09-04 | vottojo01 | 2023-10-01 | vottj001 | 2016.0 | CIN | NL | 20000000.0 | 0.375235 |

<p>126 rows × 53 columns</p>
</div>

There are 126 rows available providing records of this instance. Joey
Votto and Jose Abreu were two players in 2020 and 2021 that had a salary
of over 3650000 and an OBP of over 0.35

Is there a strong correlation between the number of at bats and on base
percentage?

``` python
sns.relplot(data = f_ab, x = 'AB', y ='OBP')
plt.show()
```

![](README_files/figure-commonmark/cell-9-output-1.png)

There is a stronger correlation here between AB and OBP compared to
salary. This makes sense as likely the more at bats for a player the
better chance they have to raise their OBP

Is there a strong correlation between RBIs and OBP?

``` python
sns.relplot(data = f_ab, x = 'RBI', y ='OBP')
plt.show()
```

![](README_files/figure-commonmark/cell-10-output-1.png)

The correlation between RBI’s and OBP appears to be positive just as
AB’s and OBP. RBI’s do not necessarily correlate to OBP as a player can
record an RBI and still be out on the play, but there are many
scenarios(double - leads to a run score) that a player gets on base and
scores runs in the process.

What is the correlation between OBP and multiple variables: HR, H, BB,
HBP, RBI, AB, games batting, and salary? Which variables have the
highest correlation?

``` python
fit = smf.ols(formula='OBP ~ HR + H + BB + HBP + RBI + AB + G_batting + salary', data = f_ab).fit()
fit.summary()
```

|                   |                  |                     |           |
|-------------------|------------------|---------------------|-----------|
| Dep. Variable:    | OBP              | R-squared:          | 0.776     |
| Model:            | OLS              | Adj. R-squared:     | 0.772     |
| Method:           | Least Squares    | F-statistic:        | 222.8     |
| Date:             | Fri, 05 Dec 2025 | Prob (F-statistic): | 8.28e-162 |
| Time:             | 16:08:49         | Log-Likelihood:     | 1274.5    |
| No. Observations: | 524              | AIC:                | -2531.    |
| Df Residuals:     | 515              | BIC:                | -2493.    |
| Df Model:         | 8                |                     |           |
| Covariance Type:  | nonrobust        |                     |           |

OLS Regression Results

|           |           |          |         |          |           |          |
|-----------|-----------|----------|---------|----------|-----------|----------|
|           | coef      | std err  | t       | P\>\|t\| | \[0.025   | 0.975\]  |
| Intercept | 0.2914    | 0.003    | 91.924  | 0.000    | 0.285     | 0.298    |
| HR        | -0.0005   | 0.000    | -1.909  | 0.057    | -0.001    | 1.54e-05 |
| H         | 0.0022    | 9.86e-05 | 22.422  | 0.000    | 0.002     | 0.002    |
| BB        | 0.0016    | 7.76e-05 | 20.773  | 0.000    | 0.001     | 0.002    |
| HBP       | 0.0016    | 0.000    | 5.502   | 0.000    | 0.001     | 0.002    |
| RBI       | 9.433e-05 | 0.000    | 0.715   | 0.475    | -0.000    | 0.000    |
| AB        | -0.0007   | 3.98e-05 | -18.656 | 0.000    | -0.001    | -0.001   |
| G_batting | 0.0003    | 0.000    | 2.570   | 0.010    | 6.15e-05  | 0.000    |
| salary    | 2.103e-10 | 2.31e-10 | 0.910   | 0.363    | -2.44e-10 | 6.64e-10 |

|                |         |                   |          |
|----------------|---------|-------------------|----------|
| Omnibus:       | 142.315 | Durbin-Watson:    | 0.593    |
| Prob(Omnibus): | 0.000   | Jarque-Bera (JB): | 441.289  |
| Skew:          | -1.273  | Prob(JB):         | 1.50e-96 |
| Kurtosis:      | 6.706   | Cond. No.         | 1.63e+07 |

<br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.63e+07. This might indicate that there are<br/>strong multicollinearity or other numerical problems.

There are enough variables contained in this regression to display a
positive correlation. It is important to note the multicollinearity that
occurs here as stats like Hits, BB, HBP, and AB all are included in the
OBP formula. However, the goal was to research which stats impacted OBP
more than others, even those included within the OBP formula.
