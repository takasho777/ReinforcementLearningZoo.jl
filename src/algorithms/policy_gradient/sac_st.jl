export SACFixedPolicyST, SACAutoPolicyST, SACPolicyNetworkST
# My own implementation of SAC with fixed temperature. Can be used with multi-thread env.

using Random
using Flux
using Flux.Losses: mse
using Distributions: Normal, logpdf

struct SACRandomPolicy{T<:AbstractFloat} <: AbstractPolicy
    rng::AbstractRNG
    na::Int  # action size
end
function (p::SACRandomPolicy{T})(env::AbstractEnv) where T
    action_space = get_actions(env)
    if action_space isa ContinuousSpace  # scalar action
        return rand(p.rng,T)*2 - one(T)
    else
        return [rand(p.rng, T)*2 - one(T) for i = 1:p.na]
    end
end
function (p::SACRandomPolicy{T})(env::MultiThreadEnv) where T
    a = Array{T, 2}(undef, p.na, length(env))
    for j = 1:length(env)
        for i = 1:p.na
            a[i, j] = rand(p.rng, T)*2 - one(T)
        end
    end
    a
end

# Define SAC Actor
struct SACPolicyNetworkST
    pre::Chain
    mean::Chain
    σ::Chain
end
Flux.@functor SACPolicyNetworkST
function (m::SACPolicyNetworkST)(state)
    x = m.pre(state)
    m.mean(x), m.σ(x) .|> softplus  # NOTE I changed this with softplus
end

# SACFixedPolicyST is the original SAC that uses fixed temparature parameter α.
mutable struct SACFixedPolicyST{
    BA<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    P,
    R<:AbstractRNG,
} <: AbstractPolicy

    policy::BA
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    γ::Float32
    ρ::Float32
    α::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_every::Int
    step::Int
    rng::R
    # for logging
    q1_loss::Float32
    q2_loss::Float32
    policy_loss::Float32
    entropy::Float32
    q_est::Float32
end

"""
    SACFixedPolicyST(;kwargs...)

# Keyword arguments

- `policy`,
- `qnetwork1`,
- `qnetwork2`,
- `target_qnetwork1`,
- `target_qnetwork2`,
- `start_policy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `α = 0.2f0`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_every = 50`,
- `step = 0`,
- `rng = Random.GLOBAL_RNG`,
- `q1_loss = 0f0`,
- `q2_loss = 0f0`,
- `policy_loss = 0f0`,
- `entropy = 0f0`,
- `q_est = 0f0`,
"""
function SACFixedPolicyST(;
    policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1,
    target_qnetwork2,
    start_policy,
    γ = 0.99f0,
    ρ = 0.995f0,
    α = 0.2f0,
    batch_size = 32,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    step = 0,
    rng = Random.GLOBAL_RNG,
    q1_loss = 0f0,
    q2_loss = 0f0,
    policy_loss = 0f0,
    entropy = 0f0,
    q_est = 0f0,
)
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    SACFixedPolicyST(
        policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        γ,
        ρ,
        α,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_every,
        step,
        rng,
        q1_loss,
        q2_loss,
        policy_loss,
        entropy,
        q_est,
    )
end


# SACAutoPolicyST is the version of SAC that uses automatic temparature adjustment. The only difference is weather the temparature is learned or not.
mutable struct SACAutoPolicyST{
    BA<:NeuralNetworkApproximator,
    BC1<:NeuralNetworkApproximator,
    BC2<:NeuralNetworkApproximator,
    P,
    R<:AbstractRNG,
} <: AbstractPolicy

    policy::BA
    qnetwork1::BC1
    qnetwork2::BC2
    target_qnetwork1::BC1
    target_qnetwork2::BC2
    target_entropy::Float32
    γ::Float32
    ρ::Float32
    log_α::Vector{Float32}
    α_optimizer::ADAM
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_every::Int
    step::Int
    rng::R
    # for logging
    q1_loss::Float32
    q2_loss::Float32
    policy_loss::Float32
    entropy::Float32
    q_est::Float32
end

"""
    SACAutoPolicyST(;kwargs...)

# Keyword arguments

- `policy`,
- `qnetwork1`,
- `qnetwork2`,
- `target_qnetwork1`,
- `target_qnetwork2`,
- `start_policy`,
- `target_entropy`,
- `γ = 0.99f0`,
- `ρ = 0.995f0`,
- `log_α = Vector{Float32}(log(1f0))`,  I'm not sure how to initiallize this. Need to check the original implementation.
- `α_optimizer = ADAM(3e-4)`,
- `batch_size = 32`,
- `start_steps = 10000`,
- `update_after = 1000`,
- `update_every = 50`,
- `step = 0`,
- `rng = Random.GLOBAL_RNG`,
- `q1_loss = 0f0`,
- `q2_loss = 0f0`,
- `policy_loss = 0f0`,
- `entropy = 0f0`,
- `q_est = 0f0`,
"""
function SACAutoPolicyST(;
    policy,
    qnetwork1,
    qnetwork2,
    target_qnetwork1,
    target_qnetwork2,
    start_policy,
    target_entropy,
    γ = 0.99f0,
    ρ = 0.995f0,
    log_α = [log(0.2f0)],
    α_optimizer = ADAM(3e-4),
    batch_size = 32,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    step = 0,
    rng = Random.GLOBAL_RNG,
    q1_loss = 0f0,
    q2_loss = 0f0,
    policy_loss = 0f0,
    entropy = 0f0,
    q_est = 0f0,
)
    copyto!(qnetwork1, target_qnetwork1)  # force sync
    copyto!(qnetwork2, target_qnetwork2)  # force sync
    SACAutoPolicyST(
        policy,
        qnetwork1,
        qnetwork2,
        target_qnetwork1,
        target_qnetwork2,
        target_entropy,
        γ,
        ρ,
        log_α,
        α_optimizer,
        batch_size,
        start_steps,
        start_policy,
        update_after,
        update_every,
        step,
        rng,
        q1_loss,
        q2_loss,
        policy_loss,
        entropy,
        q_est,
    )
end



# This is for both single env and multi-thread env
# NOTE The state is assumed to be a vector
function (p::Union{SACFixedPolicyST, SACAutoPolicyST})(env::AbstractEnv)
    if env isa MultiThreadEnv
        p.step += length(env.envs)
    else
        p.step += 1
    end

    if p.step <= p.start_steps
        action = p.start_policy(env)
    else
        s = get_state(env)
        action, _ = evaluate(p, s)

        # testmode:
        # if testing dont sample an action, but act deterministically by
        # taking the "mean" action
    end

    if length(action) == 1 && action isa AbstractArray
        action = action[1]
    end

    action
end


"""
This function is compatible with a multidimensional action space. It's assumed that the output of the policy network has size of (n_a, n_rollout) even if n_a=1 (scalar). This is default behavior of Dense layers in Flux.
"""
function evaluate(p::Union{SACFixedPolicyST,SACAutoPolicyST}, state)
    μ, σ = p.policy(state)
    π_dist = Normal.(μ, σ)
    z = rand.(p.rng, π_dist)
    logp_π = sum(logpdf.(π_dist, z), dims = 1)  # logp of diagonal Gaussian.
    logp_π -= sum((2.0f0 .* (log(2.0f0) .- z - softplus.(-2.0f0 * z))), dims = 1)  # additional term to account for Jacobian (-sum(log(1-tanh^2(z))))
    return tanh.(z), logp_π
end

const SARTS = (:state, :action, :reward, :terminal, :next_state)

function sample_batch(rng, traj, batch_size)
    if ndims(traj[:terminal]) == 1  # then assume single env
        n_rollout, = size(traj[:terminal])
        inds = rand(rng, 1:n_rollout, batch_size) 
        s = select_last_dim(traj[:state], inds)
        a = select_last_dim(traj[:action], inds)
        r = select_last_dim(traj[:reward], inds)
        t = select_last_dim(traj[:terminal], inds)
        s′ = select_last_dim(traj[:next_state], inds)
    elseif ndims(traj[:terminal]) == 2  # then assume multi-thread env (n_env, buffer_size)
        n_env, n_rollout = size(traj[:terminal])
        inds = rand(rng, 1:n_env*n_rollout, batch_size)
        s = select_last_dim(flatten_batch(traj[:state]), inds)
        a = select_last_dim(flatten_batch(traj[:action]), inds)
        r = select_last_dim(flatten_batch(traj[:reward]), inds)
        t = select_last_dim(flatten_batch(traj[:terminal]), inds)
        s′ = select_last_dim(flatten_batch(traj[:next_state]), inds)
    else
        error("dimension of the trajectory is not compatible")
    end
        
    batch = NamedTuple{SARTS}((s, a, r, t, s′))
end

function RLBase.update!(
    p::Union{SACFixedPolicyST,SACAutoPolicyST},
    traj::CircularCompactSARTSATrajectory,
)
    length(traj[:terminal]) > p.update_after || return
    p.step % p.update_every == 0 || return
    # NOTE The following loop is to fix the ratio of env steps and grad steps 1. (following spinningup implementation)
    for _ in 1:p.update_every
        batch = sample_batch(p.rng, traj, p.batch_size)
        update!(p, batch)
    end
end

# gradient step for fixed-temparature SAC
function RLBase.update!(p::SACFixedPolicyST, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = batch

    γ, ρ, α = p.γ, p.ρ, p.α

    # !!! we have several assumptions here, need revisit when we have more complex environments

    # This works for vector action
    a′, log_π = evaluate(p, s′)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec((q′ .- α .* log_π))

    # Train Q Networks
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
        q1 = p.qnetwork1(q_input) |> vec
        loss = mse(q1, y)
        ignore() do
            p.q1_loss = loss
        end
        loss
    end
    update!(p.qnetwork1, q_grad_1)
    q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
        q2 = p.qnetwork1(q_input) |> vec
        loss = mse(q2, y)
        ignore() do
            p.q2_loss = loss
        end
        loss
    end
    update!(p.qnetwork2, q_grad_2)

    # Train Policy
    p_grad = gradient(Flux.params(p.policy)) do
        a, log_π = evaluate(p, s)
        q_input = vcat(s, a)
        q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
        loss = mean(α .* log_π .- q)
        ignore() do
            p.entropy = mean(log_π)
            p.policy_loss = loss
            p.q_est = mean(q)
        end
        loss
    end
    update!(p.policy, p_grad)

    # polyak averaging
    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end



# gradient step for SAC with automatic temparature adjustment
function RLBase.update!(p::SACAutoPolicyST, batch::NamedTuple{SARTS})
    s, a, r, t, s′ = batch

    γ, ρ, log_α = p.γ, p.ρ, p.log_α

    # !!! we have several assumptions here, need revisit when we have more complex environments

    # This works for vector action
    a′, log_π = evaluate(p, s′)
    q′_input = vcat(s′, a′)
    q′ = min.(p.target_qnetwork1(q′_input), p.target_qnetwork2(q′_input))

    y = r .+ γ .* (1 .- t) .* vec((q′ .- exp.(log_α) .* log_π))

    # Train Q Networks
    q_input = vcat(s, a)

    q_grad_1 = gradient(Flux.params(p.qnetwork1)) do
        q1 = p.qnetwork1(q_input) |> vec
        loss = mse(q1, y)
        ignore() do
            p.q1_loss = loss
        end
        loss
    end
    update!(p.qnetwork1, q_grad_1)
    q_grad_2 = gradient(Flux.params(p.qnetwork2)) do
        q2 = p.qnetwork1(q_input) |> vec
        loss = mse(q2, y)
        ignore() do
            p.q2_loss = loss
        end
        loss
    end
    update!(p.qnetwork2, q_grad_2)

    # Train Policy
    p_grad = gradient(Flux.params(p.policy)) do
        a, log_π = evaluate(p, s)
        q_input = vcat(s, a)
        q = min.(p.qnetwork1(q_input), p.qnetwork2(q_input))
        loss = mean(exp.(log_α) .* log_π .- q)
        ignore() do
            p.entropy = -mean(log_π)
            p.policy_loss = loss
            p.q_est = mean(q)
        end
        loss
    end
    update!(p.policy, p_grad)
    
    # Update temparature
    # operate with log_α to make sure α is positive.
    α_grad = gradient(params(log_α)) do
        a, log_π = evaluate(p, s)
        loss = mean(-exp.(log_α) .* (log_π .+ p.target_entropy))
    end
    Flux.Optimise.update!(p.α_optimizer, params(log_α), α_grad) 

    # polyak averaging
    for (dest, src) in zip(
        Flux.params([p.target_qnetwork1, p.target_qnetwork2]),
        Flux.params([p.qnetwork1, p.qnetwork2]),
    )
        dest .= ρ .* dest .+ (1 - ρ) .* src
    end
end
