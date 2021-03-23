
function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:SACFixedST},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_SACFixedST_Pendulum_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    action_space = get_actions(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(get_state(inner_env))

    env = inner_env |> ActionTransformedEnv(x -> low + (x + 1) * 0.5 * (high - low))
    init = glorot_uniform(rng)

    init_policy_last(rng) = (dims...) -> Flux.glorot_uniform(rng, dims...)/100
    softplus_inv(y) = log(exp(y)-1)
    σ0 = 1f0

    # tanh is automatically applied to the output of network as tanh(μ+σz) where z ∼ N(0, 1)
    create_policy_net() = NeuralNetworkApproximator(
        model = SACPolicyNetworkST(
            Chain(
                Dense(ns, 64, relu; initW=init), 
                Dense(64, 64, relu; initW=init),
            ),
            Chain(
                Dense(64, 1, initW = init_policy_last(rng)),
            ),
            Chain(
                Dense(64, 1, initW = init_policy_last(rng)),
                x -> x .+ softplus_inv(σ0),
            ),
        ),
        optimizer = ADAM(0.003),
    )

    create_q_net() = NeuralNetworkApproximator(
        model = Chain(
            Dense(ns + 1, 64, relu; initW = init),
            Dense(64, 64, relu; initW = init),
            Dense(64, 1; initW = init),
        ),
        optimizer = ADAM(0.003),
    )

    update_every = 36
    agent = Agent(
        policy = SACFixedPolicyST(
            policy = create_policy_net(),
            qnetwork1 = create_q_net(),
            qnetwork2 = create_q_net(),
            target_qnetwork1 = create_q_net(),
            target_qnetwork2 = create_q_net(),
            γ = 0.99f0,
            ρ = 0.995f0,
            α = 0.1f0,
            batch_size = 64,
            start_steps = 1000,
            start_policy = RandomPolicy(ContinuousSpace(-1.0, 1.0); rng = rng),
            update_after = 1000,
            update_every = update_every,
            rng = rng,
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 100_000,
            state_type = Float32,
            state_size = (ns,),
            action_type = Float32,
            action_size = (1, ),
        ),
    )

    stop_condition = StopAfterStep(100_000)
    total_reward_per_episode = TotalRewardPerEpisode()
    hook = ComposedHook(
        total_reward_per_episode,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" return_at_terminal = total_reward_per_episode.rewards[end] log_step_increment=0
            end
        end,
        DoEveryNStep(update_every, 0) do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    q1_loss = agent.policy.q1_loss,
                    q2_loss = agent.policy.q2_loss,
                    policy_loss = agent.policy.policy_loss,
                    entropy = agent.policy.entropy,
                    q_est = agent.policy.q_est,
                    log_step_increment = update_every,
                )
            end
        end
    )

    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# Play Pendulum with SAC", save_dir),
    )
end


function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:SACFixedST},
    ::Val{:Pendulum},
    ::Val{:MultiThread},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_SACFixedST_Pendulum_MultiThread_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    action_space = get_actions(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(get_state(inner_env))
    
    N_ENV = 16
    env = MultiThreadEnv([
        PendulumEnv(; T = Float32, rng = MersenneTwister(hash(seed + i))) |> ActionTransformedEnv(x -> low + (x + 1) * 0.5 * (high - low))  for i in 1:N_ENV
    ])


    init = glorot_uniform(rng)

    init_policy_last(rng) = (dims...) -> Flux.glorot_uniform(rng, dims...)/100
    softplus_inv(y) = log(exp(y)-1)
    σ0 = 1f0

    # tanh is automatically applied to the output of network as tanh(μ+σz) where z ∼ N(0, 1)
    create_policy_net() = NeuralNetworkApproximator(
        model = SACPolicyNetworkST(
            Chain(
                Dense(ns, 64, relu; initW=init), 
                Dense(64, 64, relu; initW=init),
            ),
            Chain(
                Dense(64, 1, initW = init_policy_last(rng)),
            ),
            Chain(
                Dense(64, 1, initW = init_policy_last(rng)),
                x -> x .+ softplus_inv(σ0),
            ),
        ),
        optimizer = ADAM(0.003),
    )

    create_q_net() = NeuralNetworkApproximator(
        model = Chain(
            Dense(ns + 1, 64, relu; initW = init),
            Dense(64, 64, relu; initW = init),
            Dense(64, 1; initW = init),
        ),
        optimizer = ADAM(0.003),
    )

    update_every = 16
    agent = Agent(
        policy = SACFixedPolicyST(
            policy = create_policy_net(),
            qnetwork1 = create_q_net(),
            qnetwork2 = create_q_net(),
            target_qnetwork1 = create_q_net(),
            target_qnetwork2 = create_q_net(),
            γ = 0.99f0,
            ρ = 0.995f0,
            α = 0.1f0,
            batch_size = 64,
            start_steps = 1000,
            start_policy = RandomPolicy(ContinuousSpace(-1.0, 1.0); rng = rng),
            update_after = 1000,
            update_every = update_every,
            rng = rng,
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 100_000,
            state_type = Float32,
            state_size = (ns, N_ENV),
            action_type = Float32,
            action_size = (1, N_ENV),
            reward_type = Float32,
            reward_size = (N_ENV, ),
            terminal_type = Bool,
            terminal_size = (N_ENV, ),
        ),
    )

    stop_condition = StopAfterStep(100_000 ÷ N_ENV)
    total_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
    hook = ComposedHook(
        total_reward_per_episode,
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    q1_loss = agent.policy.q1_loss,
                    q2_loss = agent.policy.q2_loss,
                    policy_loss = agent.policy.policy_loss,
                    entropy = agent.policy.entropy,
                    q_est = agent.policy.q_est,
                    log_step_increment = N_ENV,
                )
                for i in 1:length(env)
                    if get_terminal(env[i])
                        @info "training" return_at_terminal=total_reward_per_episode.rewards[i][end] log_step_increment=0
                        break
                    end
                end
            end
        end
    )

    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# Play Pendulum with SAC using MultiThreadEnv", save_dir),
    )
end



# Single env, with automatic temparature control.
function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:SACAutoST},
    ::Val{:Pendulum},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    if isnothing(save_dir)
        t = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
        save_dir = joinpath(pwd(), "checkpoints", "JuliaRL_SACAutoST_Pendulum_$(t)")
    end

    lg = TBLogger(joinpath(save_dir, "tb_log"), min_level = Logging.Info)
    rng = MersenneTwister(seed)
    inner_env = PendulumEnv(T = Float32, rng = rng)
    action_space = get_actions(inner_env)
    low = action_space.low
    high = action_space.high
    ns = length(get_state(inner_env))

    env = inner_env |> ActionTransformedEnv(x -> low + (x + 1) * 0.5 * (high - low))
    init = glorot_uniform(rng)

    init_policy_last(rng) = (dims...) -> Flux.glorot_uniform(rng, dims...)/100
    softplus_inv(y) = log(exp(y)-1)
    σ0 = 1f0

    # tanh is automatically applied to the output of network as tanh(μ+σz) where z ∼ N(0, 1)
    create_policy_net() = NeuralNetworkApproximator(
        model = SACPolicyNetworkST(
            Chain(
                Dense(ns, 64, relu; initW=init), 
                Dense(64, 64, relu; initW=init),
            ),
            Chain(
                Dense(64, 1, initW = init_policy_last(rng)),
            ),
            Chain(
                Dense(64, 1, initW = init_policy_last(rng)),
                x -> x .+ softplus_inv(σ0),
            ),
        ),
        optimizer = ADAM(0.003),
    )

    create_q_net() = NeuralNetworkApproximator(
        model = Chain(
            Dense(ns + 1, 64, relu; initW = init),
            Dense(64, 64, relu; initW = init),
            Dense(64, 1; initW = init),
        ),
        optimizer = ADAM(0.003),
    )

    update_every = 36
    agent = Agent(
        policy = SACAutoPolicyST(
            policy = create_policy_net(),
            qnetwork1 = create_q_net(),
            qnetwork2 = create_q_net(),
            target_qnetwork1 = create_q_net(),
            target_qnetwork2 = create_q_net(),
            target_entropy = -1f0,
            γ = 0.99f0,
            ρ = 0.995f0,
            log_α = [log(1f0)],
            α_optimizer = ADAM(3e-4),
            batch_size = 64,
            start_steps = 1000,
            start_policy = RandomPolicy(ContinuousSpace(-1.0f0, 1.0f0); rng = rng),  # output of the start policy is expected to be same as the output of the network.
            update_after = 1000,
            update_every = update_every,
            rng = rng,
        ),
        trajectory = CircularCompactSARTSATrajectory(
            capacity = 100_000,
            state_type = Float32,
            state_size = (ns,),
            action_type = Float32,
            action_size = (1, ),
        ),
    )

    stop_condition = StopAfterStep(100_000)
    total_reward_per_episode = TotalRewardPerEpisode()
    hook = ComposedHook(
        total_reward_per_episode,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" return_at_terminal = total_reward_per_episode.rewards[end] log_step_increment=0
            end
        end,
        DoEveryNStep(update_every, 0) do t, agent, env
            with_logger(lg) do
                @info(
                    "training",
                    q1_loss = agent.policy.q1_loss,
                    q2_loss = agent.policy.q2_loss,
                    policy_loss = agent.policy.policy_loss,
                    entropy = agent.policy.entropy,
                    q_est = agent.policy.q_est,
                    alpha = exp(agent.policy.log_α[1]),
                    log_step_increment = update_every,
                )
            end
        end
    )

    Experiment(
        agent,
        env,
        stop_condition,
        hook,
        Description("# Play Pendulum with SAC", save_dir),
    )
end
