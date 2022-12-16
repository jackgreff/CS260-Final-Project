import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
from scipy.special import expit

def main():
    np.set_printoptions(precision=3, suppress=True)
    pitchers_data = np.genfromtxt("pitch_arsenals.csv", skip_header=1,
                                    usecols=[2, 3], missing_values="",
                                    delimiter=",")
    #drop the NA values
    pitchers_data = pitchers_data[~np.isnan(pitchers_data[:, 0])]

    velocity = pitchers_data[:, 0]
    injury = pitchers_data[:, 1]  # injury or not?

    #notice the`value` here. We explain why below.
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, tau=0.001, testval=0)
        alpha = pm.Normal("alpha", mu=0, tau=0.001, testval=0)
        p = pm.Deterministic("p", 1.0/(1. + tt.exp(beta*velocity + alpha))) #logistic
        observed = pm.Bernoulli("bernoulli_obs", p, observed=injury)

        # Mysterious code to be explained in Chapter 3
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(300000, step=step, start=start)
        burned_trace = trace[10000::2]

        alpha_samples, beta_samples = plot_posteriors(burned_trace)
        t, p_t, mean_prob_t = plot_probabilities(alpha_samples, beta_samples, velocity, injury)
        plot_credible_intervals(t, p_t, mean_prob_t, velocity, injury)

def plot_posteriors(burned_trace):
    alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
    beta_samples = burned_trace["beta"][:, None]
    #plt.subplot(211)
    #plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
    #plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
             #label=r"posterior of $\beta$", color="#7A68A6", density=True)
    #plt.legend()
    #plt.subplot(212)
    #plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
             #label=r"posterior of $\alpha$", color="#A60628", density=True)
    #plt.legend();
    #plt.savefig('plot1')
    return alpha_samples, beta_samples

def plot_probabilities(alpha_samples, beta_samples, velocity, injury):
    t = np.linspace(velocity.min() - 5, velocity.max()+5, 50)[:, None]
    p_t = logistic(t.T, beta_samples, alpha_samples)
    mean_prob_t = p_t.mean(axis=0)
    #p_t = expit(np.dot(t.T,beta_samples) + alpha_samples)
    #plt.plot(t, mean_prob_t, lw=3, label="average posterior \nprobability of injury")
    #plt.plot(t, p_t[0, :], ls="--", label="realization from posterior")
    #plt.plot(t, p_t[-2, :], ls="--", label="realization from posterior")
    #plt.scatter(velocity, injury, color="k", s=50, alpha=0.5)
    #plt.title("Posterior expected value of probability of injury; plus realizations")
    #plt.legend(loc="lower left")
    #plt.ylim(-0.1, 1.1)
    #plt.xlim(t.min(), t.max())
    #plt.ylabel("probability")
    #plt.xlabel("velocity");
    mean_prob_t = p_t.mean(axis=0)
    #plt.savefig('plot2')
    return t, p_t, mean_prob_t

def plot_credible_intervals(t, p_t, mean_prob_t, velocity, injury):
    # vectorized bottom and top 2.5% quantiles for credible intervals
    qs = mquantiles(p_t, [0.025, 0.975], axis=0)
    plt.fill_between(t[:, 0], *qs, alpha=0.7,
                     color="#7A68A6")

    plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)

    plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
             label="average posterior \nprobability of injury")

    plt.xlim(t.min(), t.max())
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="lower left")
    plt.scatter(velocity, injury, color="k", s=50, alpha=0.5)
    plt.xlabel("velocity, $t$")

    plt.ylabel("probability estimate")
    plt.title("Posterior probability estimates given velocity. $t$");
    plt.savefig('plot3')

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

if __name__ == "__main__":
    main()