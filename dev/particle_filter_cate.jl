function particle_filter(b_samples::Vector,o_truth,Δ,num_cate_vars)
    """
    Particle filter for categorical variables
    :param b_samples: samples from the belief (states)
    :param o_truth: real observations
    :param Δ: the mismatch between real and simulated observations (likelihood)
    :param num_cate_vars: total number of levels of the categorical variable
    :return: updated PMF
    """
    𝐰 = Δ.(b_samples,o_truth)
    𝒞 = Categorical(normalize(𝐰, 1))
    prob_res = Dict()
    for (idx, ele) in enumerate(b_samples)
        if get(prob_res,ele,"false") == "false"
            prob_res[ele] = 𝒞.p[idx]
        else
            prob_res[ele] += 𝒞.p[idx]
        end
    end
    return_prob_res = repeat([0.0],num_cate_vars)
    for i in 1:num_cate_vars
        if get(prob_res,i,"false") != "false"
            return_prob_res[i] = prob_res[i]
        end
    end
    return return_prob_res
end
