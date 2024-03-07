import pyblp
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from os import listdir

# TODO: Questions to ask 
# 1. pyblp - where can we find all the naming conventions

# Set directory
DATADIR ='/home/econ87/Documents/Research/Workshops/Demand-Estimation/Exercises/Data' 
listdir(DATADIR)

# Set packages config
pyblp.options.digits = 3
pyblp.options.verbose = False
pd.options.display.precision = 3
pd.options.display.max_columns = 50

# Load data
prod_df = pd.read_csv(f'{DATADIR}/products.csv')

## 1. Describe data
prod_df.head()
# prod_df.sample()
prod_df.info()
prod_df.describe()

## 2. Compute market shares - s_jt = q_jt/M_t
## Need to identify the outside option
## Assume that M_t = pop_t x 90
prod_df = prod_df.assign(market_size = prod_df.city_population * 90)
prod_df = prod_df.assign(market_share = prod_df.servings_sold / prod_df.market_size)

# Compute market share of outside option
prod_df = prod_df.assign(
        outside_share = 1-prod_df.groupby(
            ['market'])['market_share'].transform('sum'))

## Create deltas - log ratio shares log(sjt/s0jt) = δjt
prod_df = prod_df.assign(
        logit_delta = np.log(prod_df.market_share/prod_df.outside_share))

## 3. Estimate logit model using OLS (by statsmodels)
mod = smf.ols(formula = 'logit_delta ~ 1 + mushy + price_per_serving', data = prod_df)
res = mod.fit(cov_type='HC0')
print(res.summary())

# Re-intereting mushy coeff in terms of money metric
round(res.params['mushy'] / res.params['price_per_serving'], 3)


## 4. Estimate logit model using pyblp (requires specific var names)
prod_df = prod_df.rename(columns = {
    'market': 'market_ids',
    'product': 'product_ids',
    'market_share': 'shares',
    'price_per_serving': 'prices',
    })

# Assume exogenous prices and instrument prices with prices
prod_df = prod_df.assign(demand_instruments0 = prod_df.prices)
ols_problem = pyblp.Problem(pyblp.Formulation('1+mushy+prices'), prod_df)
print(ols_problem)
ols_problem.products.X1
ols_problem.products.ZD

# Solve 
ols_pyblp_res = ols_problem.solve(method='1s')
print(ols_pyblp_res)


## 5. Add FEs: market id and product id (drop mushy since fixed)
## Use the absorb option to absorb (demean) the fixed effects
ols_problem_fes = pyblp.Problem(
        pyblp.Formulation('1+prices',
        absorb = 'C(market_ids) + C(product_ids)'),
        prod_df)
print(ols_problem_fes)

# Solve 
ols_fes_pyblp_res = ols_problem_fes.solve(method='1s')
print(ols_fes_pyblp_res)

## 6. IV Estimation: Add an instrument for price
##   First-stage - test the relevance condition
first_stage = smf.ols(
        formula = 'prices ~ 1 + mushy + price_instrument + C(market_ids) + C(product_ids)',
        data = prod_df)
fir_res = first_stage.fit(cov_type='HC0')
print(fir_res.summary())

# Set price_instrument as demand_instruments0 in order for pyblp to pick it up
prod_df = prod_df.assign(demand_instruments0 = prod_df.price_instrument)
iv_problem = pyblp.Problem(
        pyblp.Formulation(
            '1 + prices',
            absorb = 'C(market_ids) + C(product_ids)'),
        prod_df)
print(iv_problem)

# Solve and print the results
iv_pyblp_res = iv_problem.solve(method='1s')
print(iv_pyblp_res)


## 7. Counterfactual analysis - half the price of product F1B04 in market C01Q2
# Create new dataset with just market C01Q2
counterfactual_df = prod_df.query('market_ids=="C01Q2"').copy()
# Half the price of F1B04
counterfactual_df['new_prices'] = counterfactual_df['prices']
counterfactual_df.loc[
        counterfactual_df['product_ids']=="F1B04",
        "new_prices"] = counterfactual_df.new_prices/2

# Compute counterfactual shares shares
counterfactual_df['new_shares'] = iv_pyblp_res.compute_shares(
        market_id='C01Q2',
       prices=counterfactual_df['new_prices'])

counterfactual_df['perc_change_share'] = (
    (counterfactual_df['new_shares'] - counterfactual_df['shares']) 
    / counterfactual_df['shares']
        )
        

## 8. Compute demand elasticities
pd.DataFrame(iv_pyblp_res.compute_elasticities(name='prices', market_id = "C01Q2"))

## S1. Try different standard errors
prod_df = prod_df.assign(clustering_ids = prod_df.product_ids)
iv_problem_cluster = pyblp.Problem(
        pyblp.Formulation(
            '1 + prices',
            absorb = 'C(market_ids) + C(product_ids)'),
        prod_df)

iv_pyblp_res_cluster = iv_problem_cluster.solve(method='1s', se_type="clustered")


pd.DataFrame(index=iv_pyblp_res.beta_labels, data={
    ("Estimates", "Unclustered"): iv_pyblp_res.beta.flat,
    ("SEs", "Unclustered"): iv_pyblp_res.beta_se.flat,
    ("Estimates", "Clustered"): iv_pyblp_res_cluster.beta.flat,
    ("SEs", "Clustered"): iv_pyblp_res_cluster.beta_se.flat,
})

## S2. Confidence Internals for counterfactual price change via Bootstrap
counterfactual_market = "C01Q2"

bootstrap_cluster_CI_res = iv_pyblp_res_cluster.bootstrap(draws=100, seed = 1)
print(bootstrap_cluster_CI_res)

# Bootstrapped shares for the counterfactual market. Their first axis indexes draws.
bootstrap_shares = bootstrap_cluster_CI_res.bootstrapped_shares[
        :, prod_df['market_ids'] == counterfactual_market]
print(bootstrap_shares.shape)



# Replicaate the bootstrapped prices (one for each draw) and bootrap the counterfactual
bootstrap_new_prices = np.tile(counterfactual_df['new_prices'].values, (100, 1))
bootstrap_new_shares = bootstrap_cluster_CI_res.compute_shares(
        market_id=counterfactual_market, prices=bootstrap_new_prices)
bootstrap_changes = 100 * (bootstrap_new_shares - bootstrap_shares) / bootstrap_shares

# Compute the 95% CI for each dchange
counterfactual_df['iv_change_lb'] = np.squeeze(np.percentile(bootstrap_changes, 2.5, axis=0))
counterfactual_df['iv_change_ub'] = np.squeeze(np.percentile(bootstrap_changes, 97.5, axis=0))
counterfactual_df


## S3. Impute marginal costs from pricing optimality
# Define firm id
prod_df = prod_df.assign(firm_ids=prod_df.product_ids.str[:2])

firm_problem = (
        pyblp.Problem(
        pyblp.Formulation(
            '1+prices',
            absorb = 'C(market_ids) + C(product_ids)'),
        prod_df)
        )

firm_results = firm_problem.solve(method='1s', se_type='clustered')

# Impute marginal costs from the pricing  optimality condition and compare
# with prices; we can also comptue markups and profits
prod_df['costs'] = firm_results.compute_costs()
prod_df['profit_per_serving'] = prod_df['prices'] - prod_df['costs']

prod_df['markups'] = prod_df['profit_per_serving'] / prod_df['costs']
prod_df[['prices', 'costs', 'profit_per_serving', 'markups']].describe()
