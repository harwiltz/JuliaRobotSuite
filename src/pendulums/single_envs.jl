export AbstractStochasticCartPoleEnv, CartPoleTask, CartPoleBalance, CartPoleSwingup
export CARTPOLE_BALANCE, CARTPOLE_SWINGUP
export StochasticCartPoleEnv

using IntervalSets
using Random
using Distributions
using DifferentialEquations
using LinearAlgebra
using Plots

import ReinforcementLearningBase as RLBase

const τ = 5e-3

abstract type AbstractStochasticCartPoleEnv <: RLBase.AbstractEnv end

abstract type CartPoleTask end
struct CartPoleBalance <: CartPoleTask end
struct CartPoleSwingup <: CartPoleTask end

const CARTPOLE_BALANCE = CartPoleBalance()
const CARTPOLE_SWINGUP = CartPoleSwingup()

pendulum(env::AbstractStochasticCartPoleEnv) = nothing
poisson_λ(env::AbstractStochasticCartPoleEnv) = nothing
cur_time(env::AbstractStochasticCartPoleEnv) = nothing
max_time(env::AbstractStochasticCartPoleEnv) = nothing
update_state!(env::AbstractStochasticCartPoleEnv, state::InvertedPendulumState) = nothing

plot(env::AbstractStochasticCartPoleEnv) = plot(pendulum(env))

RLBase.action_space(env::AbstractStochasticCartPoleEnv) = nothing
RLBase.state_space(env::AbstractStochasticCartPoleEnv) =
    RLBase.Space(ClosedInterval{Float32}[-5..5,
                                         typemin(Float32)..typemax(Float32),
                                         0f0..2π,
                                         typemin(Float32)..typemax(Float32)])
RLBase.reward(env::AbstractStochasticCartPoleEnv) = nothing
RLBase.is_terminated(env::AbstractStochasticCartPoleEnv) = nothing
RLBase.state(env::AbstractStochasticCartPoleEnv) = state_vec(pendulum(env))
RLBase.reset!(env::AbstractStochasticCartPoleEnv) = nothing

mutable struct StochasticCartPoleEnv{Ta <: CartPoleTask, F <: Real, A} <: AbstractStochasticCartPoleEnv
    pendulum::InvertedPendulum{F}
    λ::F
    Nₐ::A
    reward::F
    is_terminated::Bool
    T::F
    t::F
    dt::F
end

pendulum(env::StochasticCartPoleEnv) = env.pendulum
poisson_λ(env::StochasticCartPoleEnv) = env.λ
cur_time(env::StochasticCartPoleEnv) = env.t
max_time(env::StochasticCartPoleEnv) = env.T

function update_state!(
    env::StochasticCartPoleEnv{Ta,F,A},
    state::InvertedPendulumState{F}
) where {Ta,F<:Real,A}
    env.pendulum.state = state
end

RLBase.action_space(env::StochasticCartPoleEnv{Ta,F,A}) where {Ta,F,A<:Nothing} = ClosedInterval{F}(-1, 1)
RLBase.action_space(env::StochasticCartPoleEnv{Ta,F,A}) where {Ta,F,A<:Int} = Base.OneTo(env.Nₐ)
RLBase.reward(env::StochasticCartPoleEnv) = env.reward
RLBase.is_terminated(env::StochasticCartPoleEnv) = env.is_terminated

reset_config(env::StochasticCartPoleEnv{Ta,F,A}) where {Ta<:CartPoleBalance,F,A} =
    InvertedPendulum(CartPoleConfiguration(F = F))

reset_config(env::StochasticCartPoleEnv{Ta,F,A}) where {Ta<:CartPoleSwingup,F,A} =
    InvertedPendulum(CartPoleSwingupConfiguration(F = F))

function RLBase.reset!(env::StochasticCartPoleEnv{Ta,F,A}) where {Ta,F<:Real,A}
    env.reward = 0
    env.is_terminated = false
    env.t = F(0)
    env.dt = F(0)
    env.pendulum = reset_config(env)
end

function StochasticCartPoleEnv(
    task::Ta = CARTPOLE_BALANCE;
    F = Float32,
    λ = 4,
    Nₐ = 2,
    T = 10
) where Ta <: CartPoleTask
    pendulum = InvertedPendulum(CartPoleConfiguration(F = F))
    StochasticCartPoleEnv{Ta,F,Int}(pendulum, F(λ), Nₐ, F(0), false, F(T), F(0), F(0))
end

function (env::StochasticCartPoleEnv{Ta, F, A})(a::Int) where {Ta, F, A <: Int}
    a′ = LinRange(-1, 1, env.Nₐ)[a]
    env(F(a′))
end

inner_reward(env::StochasticCartPoleEnv{Ta,F,N}) where {Ta<:CartPoleBalance,F,N} = F(1)

function is_terminated(env::StochasticCartPoleEnv{Ta,F,N}) where {Ta<:CartPoleBalance,F,N}
    s = pendulum(env).state
    alive_x = ClosedInterval{F}(-3..3)
    alive_sinθ = ClosedInterval{F}(-0.5..0.5)
    alive = (s.x ∈ alive_x) && (sin(s.θ) ∈ alive_sinθ)
    !alive || (cur_time(env) >= max_time(env))
end

function inner_reward(env::StochasticCartPoleEnv{Ta,F,N}) where {Ta<:CartPoleSwingup,F,N}
    s = pendulum(env).state
    score_region = RLBase.Space(ClosedInterval{F}[-10..10, typemin(F)..typemax(F), -0.25..0.25, -π..π])
    F(1) * ([s.x, s.dx, sin(s.θ), s.dθ] ∈ score_region)
end

is_terminated(env::StochasticCartPoleEnv{Ta,F,N}) where {Ta<:CartPoleSwingup,F,N} =
    (cur_time(env) >= max_time(env))

function (env::AbstractStochasticCartPoleEnv)(a::F) where F <: Real
    cp = pendulum(env)
    λ = poisson_λ(env)
    μ, σ = dynamics(cp, a)
    T = F(τ) #F(1 + rand(Poisson(λ))) * F(τ)
    env.dt = T
    prob = SDEProblem(μ, σ, state_vec(cp), (0.0, T))
    ts, xs = solve(prob, EM(), dt=τ)
    reward = F(0)
    done = false
    alive_x = ClosedInterval{F}(-3..3)
    alive_sinθ = ClosedInterval{F}(-0.5..0.5)
    for xᵢ in eachcol(xs)
        s = InvertedPendulumState(xᵢ...)
        update_state!(env, s)
        env.t = env.t + F(τ)
        reward = reward + F(τ * inner_reward(env))
        done = is_terminated(env)
        if done
            break
        end
    end
    env.reward = reward
    env.is_terminated = done
    state(env)
end
