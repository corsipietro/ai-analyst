## System
A teaching toy model of ONE orbital shell around the Earth, with two populations: active satellites S and debris N. Satellites are launched, retire at the end of their life, and are destroyed by collisions with debris; each collision creates new debris; debris slowly falls out of the shell because of atmospheric drag. All parameter values were chosen to make the behaviour visible: they are not calibrated on real orbits.

## Variables and units
- t: time [years], step 1 year, from year 0.
- S: active satellites [number]; N: debris objects [number]. Both are stored as real numbers.
- L: launches per year [satellites/year], the sum of the launches decided by the strategies.
- c: collisions in one year [number].

## Equations (one step = one year)
- c ~ Poisson(alpha · S · N)   (the only random part)
- S(t+1) = max(0, S + L - S/T - c)
- N(t+1) = max(0, N + f · c - N/tau)

## Parameters (same in every run)
- T = 5 years (average operational life of a satellite)
- alpha = 1e-6 collisions per year per (satellite × debris)
- f = 100 debris created by one collision
- tau = 25 years (average time before drag removes a debris)
- S(0) = 1000, N(0) = 1000

## Strategies
- greedy: launches a fixed number of satellites every year (argument: launches).

## The experiment
One greedy strategy; its number of launches per year was varied between runs, and each value was repeated with different random seeds. The results are the yearly values of S and N of every run.

## Data format
- Table `runs`: one row per run. `run_id` identifies the run, `experiment` names the experiment, `seed` is the random seed of the run, `config` is a JSON text with the run's parameters: `model.lifetime_years` = T, `model.collision_coeff` = alpha, `model.fragments_per_collision` = f, `model.debris_decay_years` = tau, `model.initial_satellites` = S(0), `model.initial_debris` = N(0), and `strategies`, a list where each strategy has a `type` and its arguments (`launches` for greedy).
- Table `series`: one row per run, year and quantity. `run_id` links to `runs`, `year` = t, `metric` names the quantity (`satellites` = S, `debris` = N), `value` is its value at that time.
