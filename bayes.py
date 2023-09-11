import functools
import operator

class Bayes:
    def __init__(self,hypothesis, priors, observations, likelihood, likelihood_dict={}):
        
        self.hyp = hypothesis
        self.og_priors = priors
        self.priors = priors
        self.obs = observations
        self.likelihood_list = likelihood

        
        self.priors_dict = dict(zip(hypothesis, priors))
        # Accessing likelihood is hyp string + Observation string. 
        # Bowl1 : chocolate: 15/50, vanilla: 35/50 Bowl2: chocolate: 30/50, vanilla 20/50
        # Make dictionary of dictionaries. This ensures that we only need the names to get the likelihood. 
        # Avoids unnesecary medling with indices.
        self.likelihood_dict =  {}
        for i, hypo in enumerate(hypothesis):
            # create dictionary that will be the value to this hypo
            inner_dict = {}
            for j, obs in enumerate(observations):
                inner_dict[obs] = likelihood[i][j]
            self.likelihood_dict[hypo] = inner_dict            

    def likelihood(self, observation, hypothesis):
    # Translate these to their corresponding indeces, then return value from likelihood
    # Or, with dic of dic, we only need the names.
    # if string, get dic dic, if number, get index
        if isinstance(observation, str) and isinstance(hypothesis, str):
            return self.likelihood_dict[hypothesis][observation]

    def norm_constant(self, observation):
    # Call Likelihood with this observation, passing as argument all possible hypothesis. Sum
        sum = 0
        for h in self.hyp:
            sum += self.likelihood(observation, h) * self.priors_dict[h]
        return sum

    def single_posterior_update(self, observation, priors):
    # P ( H | O) for all hypothesis = P( H ) * P (O | H ) / P ( O )
    # Get Norm_const from observation P ( O )
    # For each H, get likelihood  P ( O | H ) with likelihood function
    # For each H, get prior, P ( H ). Question is, is priors here a string or a number? Probably number, as we will update priors with each observation
    # Update local prior as well
        norm = self.norm_constant(observation)
        n_priors = []
        for i, h in enumerate(self.hyp):
            likelihood = self.likelihood(observation, h)
            prior = priors[i]
            posterior = (prior * likelihood) / norm
            n_priors.append(posterior)
        self.update_priors(n_priors)
        return n_priors

    def update_priors(self, priors):
        self.priors = priors
        self.priors_dict = dict(zip(self.hyp, priors))
        
            
    
    def compute_posterior(self, observations):
    # for each observation, call single posterior update, and pass the result to the next. 
    # Done best with a fold right
        #return functools.reduce(self.single_posterior_update, observations, self.og_priors)
        acc = self.og_priors; [acc := self.single_posterior_update(x, acc) for x in observations]
        return acc
        

    
    
