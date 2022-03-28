import jax
import jax.numpy as jnp
from jax.scipy import stats
from functools import partial


class BOCD:
    def __init__(self, mu0, lambda0, lambda_data, hazard):
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.lambda_data = lambda_data
        self.hazard = hazard
        self.run_length_pred = jax.vmap(self.run_length_pred, (None, 0, 0))
    
    def run_length_pred(self, x, mean, prec):
        return stats.norm.logpdf(x, mean, jnp.sqrt(1 / prec + 1))
    
    def __call__(self, X):
        T = len(X)
        init_state = {
            "time": 0,
            "mean": jnp.zeros(T).at[0].set(self.mu0),
            "precision": jnp.zeros(T).at[0].set(self.lambda0),
            "log_pred": jnp.zeros(T),
            "suff_stat": jnp.zeros(T),
        }

        bocd_step = partial(self.__bocd_step, timesteps=jnp.arange(T))
        final_state, pred = jax.lax.scan(bocd_step, init_state, X)
        return final_state, pred
    
    def __bocd_step(self, state, xt, timesteps):
        time = state["time"]
        prec_old = state["precision"][time]
        time_select = timesteps <= time
        state["suff_stat"] = jnp.roll(state["suff_stat"], 1) + xt * time_select

        log_predictive_run_length = self.run_length_pred(xt, state["mean"], state["precision"])

        pred_new_0 = jax.nn.logsumexp(state["log_pred"] + log_predictive_run_length + jnp.log(H))
        pred_new = state["log_pred"] + log_predictive_run_length  + jnp.log((1 - H))

        state["log_pred"] = jnp.r_[pred_new_0, pred_new[:-1]]
        state["log_pred"] = state["log_pred"] - jax.nn.logsumexp(state["log_pred"])

        prec_new = self.lambda_data + prec_old
        state["time"] = state["time"] + 1
        state["precision"] = state["precision"].at[time + 1].set(prec_new)
        state["mean"] = (self.mu0 / self.lambda0 + state["suff_stat"] / self.lambda_data) / state["precision"]
        state["mean"] = state["mean"] * time_select

        return state, state["log_pred"]