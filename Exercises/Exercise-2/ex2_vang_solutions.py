import pyblp
import pandas as pd
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

## Questions
# 1. Q2 - why do we need an excluded instrument for mushy x log income
#        why is is mushy considered an endogenous variable

# Directory
# Set directory
DATADIR = "/home/econ87/Documents/Research/Workshops/Demand-Estimation/Exercises/Data"
print(listdir(DATADIR))

# Set packages config
pyblp.options.digits = 3
pyblp.options.verbose = False
pd.options.display.precision = 3
pd.options.display.max_columns = 50

# Load data
prod_df = pd.read_csv(f"{DATADIR}/products.csv")
demo_df = pd.read_csv(f"{DATADIR}/demographics.csv")
demo_df = demo_df.rename(columns = {'market': 'market_ids'})

# Create log income
demo_df["log_qty_income"] = np.log(demo_df.quarterly_income)


# Define market size from exercise 1
prod_df["market_size"] = prod_df.city_population * 90
prod_df["market_share"] = prod_df.servings_sold / prod_df.market_size
prod_df = prod_df.assign(
    outside_share=1 - prod_df.groupby(["market"])["market_share"].transform("sum")
)

prod_df.head()

#  Rename vars to b understandable by pyblp
prod_df = prod_df.rename(
    columns={
        "market": "market_ids",
        "product": "product_ids",
        "market_share": "shares",
        "price_per_serving": "prices",
        "price_instrument": "demand_instruments0",
    }
)


# Estimate logit model from ex.1
first_stage = smf.ols(
    "prices ~ 0 + demand_instruments0 + C(market_ids) + C(product_ids)", prod_df
)

first_stage_results = first_stage.fit(cov_type="HC0")
iv_problem_formulation = pyblp.Formulation(
    "0 + prices", absorb="C(market_ids)+C(product_ids)"
)

iv_problem = pyblp.Problem(iv_problem_formulation, product_data = prod_df)
iv_res = iv_problem.solve(method="1s")

# Counterfactual from ex. 1
counterfactual_mkt = "C01Q2"
counterfactual_prod = "F1B04"
counterfactual_df = prod_df.loc[prod_df.market_ids == counterfactual_mkt]
counterfactual_df = counterfactual_df.reset_index(drop=True)
counterfactual_df["new_prices"] = counterfactual_df["prices"].copy()
counterfactual_df.loc[
    counterfactual_df.product_ids == counterfactual_prod, "new_prices"
] /= 2

# Predict counterfactual based on estimated iv model
counterfactual_df["counterfactual_shares"] = iv_res.compute_shares(
    market_id=counterfactual_mkt, prices=counterfactual_df.new_prices
)

counterfactual_df["iv_change"] = (
    100
    * (counterfactual_df["counterfactual_shares"] - counterfactual_df["shares"])
    / counterfactual_df["shares"]
)


# --------------------------------------------------------------------------

## 1. Describe cross-market variation
prod_df.groupby('market_ids', as_index=False).agg(**{
    'products': ('product_ids', 'count'),
    'mushy_mean': ('mushy', 'mean'),
    'mushy_std': ('mushy', 'std'),
    'prices_mean': ('prices', 'mean'),
    'prices_std': ('prices', 'std'),
    }).describe()


demo_df.groupby(["market_ids"])["log_qty_income"].agg(
    ["count", "mean", 'std']).describe()

demo_variation_df = demo_df.groupby('market_ids', as_index=False).agg(
        **{'log_income_mean': ('log_qty_income', 'mean')})


## 2. Estimate a parmeter on mushy x log_income
# Since we have only one observation of mean income, it will be collinear with
# our FEs. That is why we focus on mushy x log_income
# We want to estimate heterogeneity so we create consumer types
NDRAWS = 10**3

agent_data = (demo_df[['market_ids', 'log_qty_income']]
              .groupby('market_ids')
              .sample(n=NDRAWS, replace=True, random_state=0)
              ).reset_index(drop=True)
# Weight each "drawn" observation -- weight is based on the size of each group
agent_data['weights'] = 1 / agent_data.groupby('market_ids').transform('size')

agent_data[['nodes0', 'nodes1', 'nodes2']] = (
        np
        .random
        .default_rng(seed = 0)
        .normal(size=(len(agent_data), 3))
        )

# Merge product and demographic data
prod_df = prod_df.merge(demo_variation_df, on = 'market_ids')

# Create new instrument
prod_df['demand_instruments1'] = prod_df.mushy * prod_df.log_income_mean

# To 
product_formulations = (
        pyblp.Formulation('0 + prices', absorb='C(market_ids) + C(product_ids)'),
        pyblp.Formulation('0 + mushy'))

agent_formulation = pyblp.Formulation('0 + log_qty_income')

mushy_problem = pyblp.Problem(
        product_formulations,
        prod_df,
        agent_formulation,
        agent_data)

print(mushy_problem)
prod_df.columns
prod_df.head

optimization = pyblp.Optimization('trust-constr', {'gtol': 1e-8, 'xtol': 1e-8})
pyblp.options.verbose = True
mushy_results = mushy_problem.solve(sigma=0, pi=1, method='1s', optimization=optimization)
pyblp.options.verbose = False


## 3. Try random starting values
pi_bounds = (-10, 10)
for seed in range(3):
    initial_pi = np.random.default_rng(seed).uniform(*pi_bounds)
    seed_res = mushy_problem.solve(sigma=0, pi=initial_pi, method='1s', optimization=optimization)
    print(f'Initial: {initial_pi}, Estimated: {seed_res.pi[0,0]}')


## 4. Price cut counterfactual
counterfactual_df['new_shares'] = (
        mushy_results
        .compute_shares(market_id=counterfactual_mkt,
                        prices=counterfactual_df['new_prices'])
        )


counterfactual_df['mushy_change'] = 100 * (counterfactual_df['new_shares'] - counterfactual_df['shares']) / counterfactual_df['shares']
counterfactual_df


