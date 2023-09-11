from bayes import Bayes

def q_1_1():

    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    # e.g. likelihood[0][1] corresponds to the likehood of Bowl1 and vanilla, or 35/50
    likelihood = [[15/50, 35/50], [30/50, 20/50]]
    b = Bayes(hypos, priors, obs, likelihood)
    obs_text = "vanilla"
    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    return p_1[0]

def q_1_2():
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    # e.g. likelihood[0][1] corresponds to the likehood of Bowl1 and vanilla, or 35/50
    likelihood = [[15/50, 35/50], [30/50, 20/50]]
    b = Bayes(hypos, priors, obs, likelihood)
    p_2 = b.compute_posterior(["chocolate", "vanilla"])
    return p_2[1]

def q_2_1():
    hypos = ["beginner", "intermediate", "advanced", "expert"]
    priors = [0.25,0.25,0.25,0.25]
    obs = ["yellow", "red", "blue", "black", "white"]
    likelihood_b = [0.05, 0.1, 0.4, 0.25, 0.2]
    likelihood_i = [0.1, 0.2, 0.4, 0.2, 0.1]
    likelihood_a = [0.2, 0.4, 0.25, 0.1, 0.05]
    likelihood_e = [0.3, 0.5, 0.125, 0.05, 0.025]
    likelihood = [likelihood_b, likelihood_i, likelihood_a, likelihood_e]
    b_archer = Bayes(hypos, priors, obs, likelihood)
    observe = ["yellow", "white", "blue", "red", "red", "blue"]
    p_after = b_archer.compute_posterior(observe)
    return p_after[1]
def q_2_2():
    hypos = ["beginner", "intermediate", "advanced", "expert"]
    priors = [0.25,0.25,0.25,0.25]
    obs = ["yellow", "red", "blue", "black", "white"]
    likelihood_b = [0.05, 0.1, 0.4, 0.25, 0.2]
    likelihood_i = [0.1, 0.2, 0.4, 0.2, 0.1]
    likelihood_a = [0.2, 0.4, 0.25, 0.1, 0.05]
    likelihood_e = [0.3, 0.5, 0.125, 0.05, 0.025]
    likelihood = [likelihood_b, likelihood_i, likelihood_a, likelihood_e]
    b_archer = Bayes(hypos, priors, obs, likelihood)
    
    observe = ["yellow", "white", "blue", "red", "red", "blue"]
    
    p_after = b_archer.compute_posterior(observe)
    # find inex of max, 
    index_of_max_prob = max(enumerate(p_after), key=lambda x: x[1])[0]
    return hypos[index_of_max_prob]
if __name__ == '__main__':
    with open("group_29.txt", "w") as text_file:
        text_file.write("%s\n" % q_1_1())
        text_file.write("%s\n" % q_1_2())
        text_file.write("%s\n" % q_2_1())
        text_file.write("%s\n" % q_2_2())


