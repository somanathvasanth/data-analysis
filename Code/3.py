#23B1032-SRIKAR NIHAL
#23B0924-B.SOMANATH VASANTH
#23B1055-ANIRUDH
#Question 3

#Task A
import numpy as np
data = np.loadtxt('3.data')
first_moment = np.mean(data)
print(f"The sample first moment is {first_moment}")
second_moment = np.mean(np.square(data))
print(f"The sample second moment is {second_moment}")


#Task B
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('3.data')
plt.hist(data,bins=100,density=True)
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Histogram of data')
plt.savefig('3b.png')


#Task C
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import binom

data = np.loadtxt('3.data')
first_moment = np.mean(data)
second_moment = np.mean(np.square(data))
mu_hat1 = first_moment
mu_hat2 = second_moment 

def equations(vars):
    n, p = vars
    eq1 = mu_hat1 - n * p
    eq2 = mu_hat2 - (n * p * (1 - p) + (n * p)**2)
    return [eq1, eq2]

initial_guess = [15, 0.5]
solution = fsolve(equations, initial_guess)
n_star = round(solution[0])
p_star = solution[1]
plt.hist(data, bins=100,density=True, label='Data')
x = np.linspace(0, n_star,n_star+1)
pmf = binom.pmf(x, n_star, p_star)
plt.plot(x, pmf, '-',color='red', label='binom(n,p)')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Histogram and Binomial PMF')
plt.legend()
plt.savefig('3c.png')


#Task D
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import gamma
import matplotlib.pyplot as plt

data = np.loadtxt('3.data')
first_moment = np.mean(data)
second_moment = np.mean(np.square(data))
mu_hat1 = first_moment
mu_hat2 = second_moment 

def equations(params):
    k, theta = params
    eq1 = mu_hat1 - k * theta  
    eq2 = mu_hat2 - (k + 1) * k * theta**2  
    return [eq1, eq2]

initial_guess = [1, 1]
k_star, theta_star = fsolve(equations, initial_guess)
print(k_star)
print(theta_star)
plt.hist(data, bins=100, density=True, label='Data')
x = np.linspace(0, max(data), 100)
plt.plot(x, gamma.pdf(x, a=k_star, scale=theta_star), '-',color ='red', label='Γ(k,theta)')
plt.xticks(np.arange(0,21,2.5))
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Histogram and Gamma PDF')
plt.legend()
plt.savefig('3d.png')


#Task E
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import binom,gamma

data = np.loadtxt('3.data')
first_moment = np.mean(data)
second_moment = np.mean(np.square(data))
mu_hat1 = first_moment
mu_hat2 = second_moment 

def equations1(vars):
    n, p = vars
    eq1 = mu_hat1 - n * p
    eq2 = mu_hat2 - (n * p * (1 - p) + (n * p)**2)
    return [eq1, eq2]

initial_guess = [15, 0.5]
solution = fsolve(equations1, initial_guess)
n_star = round(solution[0])
p_star = solution[1]

def equations2(params):
    k, theta = params
    eq1 = mu_hat1 - k * theta  
    eq2 = mu_hat2 - (k + 1) * k * theta**2  
    return [eq1, eq2]

initial_guess = [1, 1]
k_star, theta_star = fsolve(equations2, initial_guess)

data_rounded = np.rint(data).astype(int)
binom_log_likelihoods = binom.logpmf(data_rounded, n=n_star, p=p_star)
avg_binom_log_likelihood = np.mean(binom_log_likelihoods)
gamma_log_likelihoods = gamma.logpdf(data, a=k_star, scale=theta_star)
avg_gamma_log_likelihood = np.mean(gamma_log_likelihoods)
print(f"Average Log-Likelihood (Binomial): {avg_binom_log_likelihood}")
print(f"Average Log-Likelihood (Gamma): {avg_gamma_log_likelihood}")
if avg_gamma_log_likelihood > avg_binom_log_likelihood:
    print("The Gamma distribution provides a better fit.")
else:
    print("The Binomial distribution provides a better fit.")


#Task F
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt

data = np.loadtxt('3.data')

mu_hat1 = np.mean(data)
mu_hat2 = np.mean(np.square(data))
mu_hat3 = np.mean(data*(np.square(data)))
mu_hat4 = np.mean(np.square(np.square(data)))
def equations(params):
    mu1, p1, mu2, p2 = params
    
    sigma1_sq = 1
    sigma2_sq = 1
    
    eq1 = mu_hat1 - (p1 * mu1 + p2 * mu2)
    eq2 = mu_hat2 - (p1 * (sigma1_sq + (mu1**2)) + p2 * (sigma2_sq + (mu2**2)))
    eq3 = mu_hat3 - (p1 * (mu1**3 + 3 * mu1 * sigma1_sq) + p2 * ((mu2**3) + 3 * mu2 * sigma2_sq))
    eq4 = mu_hat4 - (p1 * (mu1**4 + 6 * (mu1**2) * sigma1_sq + 3 * sigma1_sq**4) + p2 * ((mu2**4) + 6 * (mu2**2) * sigma2_sq + 3 * (sigma2_sq**2)))
    
    return [eq1, eq2, eq3, eq4]

initial_guess = [6, 0.5, 6, 0.5]
mu1_star, p1_star, mu2_star, p2_star = fsolve(equations, initial_guess)

x = np.linspace(0, max(data), 100)
gmm_pdf = (p1_star * norm.pdf(x, loc=mu1_star, scale=1) +
           p2_star * norm.pdf(x, loc=mu2_star, scale=1))
plt.hist(data, bins=100, density=True, label='Data')
plt.plot(x, gmm_pdf, '-', color='r',label='GMM PDF (p1*N(mu1,1) + p2*N(mu2,1)')
plt.xticks(np.arange(0,21,2.5))
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Histogram and GMM PDF')
plt.legend()
plt.savefig('3f.png')
gmm_log_likelihoods = np.log(p1_star * norm.pdf(data, loc=mu1_star, scale=1) +
                             p2_star * norm.pdf(data, loc=mu2_star, scale=1))

avg_neg_log_likelihood = -np.mean(gmm_log_likelihoods)
print(f"Average Negative Log-Likelihood of GMM: {avg_neg_log_likelihood}")
