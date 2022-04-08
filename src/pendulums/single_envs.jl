export AbstractStochasticCartPoleEnv, BalanceStochasticCartPoleEnv, SwingupStochasticCartPoleEnv

export StochasticCartPoleEnv

using IntervalSets
using Random
using Distributions
using DifferentialEquations
using LinearAlgebra
using Plots

import ReinforcementLearningBase as RLBase

const τ = 1e-4

abstract type AbstractStochasticCartPoleEnv <: RLBase.AbstractEnv end
abstract type BalanceStochasticCartPoleEnv <: AbstractStochasticCartPoleEnv end
abstract type SwingupStochasticCartPoleEnv <: AbstractStochasticCartPoleEnv end

pendulum(env::AbstractStochasticCartPoleEnv) = nothing
poisson_λ(env::AbstractStochasticCartPoleEnv) = nothing
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

mutable struct StochasticCartPoleEnv{F <: Real, A} <: BalanceStochasticCartPoleEnv
    pendulum::InvertedPendulum{F}
    λ::F
    Nₐ::A
    reward::F
    is_terminated::Bool
end

pendulum(env::StochasticCartPoleEnv) = env.pendulum
poisson_λ(env::StochasticCartPoleEnv) = env.λ

function update_state!(env::StochasticCartPoleEnv{F, A}, state::InvertedPendulumState{F}) where {F <: Real, A}
    env.pendulum.state = state
end

RLBase.action_space(env::StochasticCartPoleEnv{F, A}) where {F, A <: Nothing} = ClosedInterval{F}(-1, 1)
RLBase.action_space(env::StochasticCartPoleEnv{F, A}) where {F, A <: Int} = Base.OneTo(env.Nₐ)
RLBase.reward(env::StochasticCartPoleEnv) = env.reward
RLBase.is_terminated(env::StochasticCartPoleEnv) = env.is_terminated
function RLBase.reset!(env::StochasticCartPoleEnv{F, A}) where {F <: Real, A}
    env.reward = 0
    env.is_terminated = false
    env.pendulum = InvertedPendulum(CartPoleConfiguration(F = F))
end

function StochasticCartPoleEnv(
    F = Float32;
    λ = 4,
    Nₐ = 2,
)
    pendulum = InvertedPendulum(CartPoleConfiguration(F = F))
    StochasticCartPoleEnv(pendulum, F(λ), Nₐ, F(0), false)
end

function (env::StochasticCartPoleEnv{F, A})(a::Int) where {F, A <: Int}
    a′ = LinRange(-1, 1, env.Nₐ)[a]
    env(F(a′))
end

function (env::BalanceStochasticCartPoleEnv)(a::F) where F <: Real
    cp = pendulum(env)
    λ = poisson_λ(env)
    μ, σ = dynamics(cp, a)
    T = F(1 + rand(Poisson(λ))) * F(τ)
    prob = SDEProblem(μ, σ, state_vec(cp), (0.0, T))
    ts, xs = solve(prob, EM(), dt=τ)
    reward = F(0)
    done = false
    for xᵢ in eachcol(xs)
        rᵢ = 1
        reward = reward + F(τ * rᵢ)
        done = !(xᵢ ∈ state_space(env))
        if done
            break
        end
    end
    env.reward = reward
    env.is_terminated = done
    update_state!(env, InvertedPendulumState(xs[end]...))
    state(env)
end
