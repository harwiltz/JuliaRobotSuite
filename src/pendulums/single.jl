using Random

import Plots.plot
import Plots.plot!
import Plots.gui

export CartPoleConfiguration, CartPoleSwingupConfiguration, InvertedPendulumState, InvertedPendulum

abstract type InvertedPendulumConfiguration end

const G = 9.81

const X_THRESH = 2.5f0
const PLOT_WIDTH = 600

struct CartPoleConfiguration{F <: Real} <: InvertedPendulumConfiguration end
struct CartPoleSwingupConfiguration{F <: Real} <: InvertedPendulumConfiguration end

CartPoleConfiguration(;F = Float32) = CartPoleConfiguration{F}()
CartPoleSwingupConfiguration(;F = Float32) = CartPoleSwingupConfiguration{F}()

struct InvertedPendulumState{F <: Real}
    x::F
    dx::F
    θ::F
    dθ::F
end

function init(::CartPoleConfiguration; R = Random.GLOBAL_RNG, F = Float32)
    x = F(0)
    dx = F(0)
    θ = F(0.2 * randn())
    dθ = F(0.2 * randn())
    InvertedPendulumState(x, dx, θ, dθ)
end

init(::CartPoleSwingupConfiguration; F = Float32, kwargs...) =
    InvertedPendulumState(F(0), F(0), F(π), F(0))

mutable struct InvertedPendulum{F <: Real} <: AbstractInvertedPendulum
    state::InvertedPendulumState{F}
    M::F
    m::F
    b::F
    l::F
    I::F
    f::F
end

function InvertedPendulum(config::InvertedPendulumConfiguration;
                          R = Random.GLOBAL_RNG, F = Float32, kwargs...)
    InvertedPendulum(init(config; R=R, F=F); kwargs...)
end

function InvertedPendulum(state::InvertedPendulumState{F};
                          M = F(0.5),
                          m = F(0.2),
                          b = F(0.1),
                          l = F(1),
                          I = F(0.006),
                          f = F(2)) where F <: Real
    InvertedPendulum(state, M, m, b, l, I, f)
end

function μ_inverted_pendulum(du, u, p, t)
    x, dx, θ, dθ = u
    du[1] = dx # x' = dx
    du[3] = dθ # θ' = dθ
    a = p[1] # action ∈ [-1, 1]
    M, m, b, l, I, f = p[2:end]
    sθ = sin(θ)
    cθ = cos(θ)
    F = a * f
    du[2] = F - 4b * dx + 4m * l * dθ^2 * sθ - 3m * G * sθ * cθ
    du[2] = du[2] / (4(M + m) - 3m * cθ^2)
    du[4] = (M + m) * G * sθ - (F - b * dx) * cθ - m * l * dθ^2 * sθ * cθ
    du[4] = du[4] / (l * 4(M + m) / 3 - m * cθ^2)
end

function σ_inverted_pendulum(du, u, p, t)
    #du = (2f0 - exp(abs(u[1]))) .* [1, 1/(2π), 1, 1/√(2π)]
    du = [1f0, Float32(1/(2π)), 1f0, 1f0]
end

function μ_dynamics(cartpole::InvertedPendulum{F}, a::F) where F <: Real
    M = cartpole.M
    m = cartpole.m
    b = cartpole.b
    l = cartpole.l
    I = cartpole.I
    f = cartpole.f
    p = [a, M, m, b, l, I, f]
    (du, u, _, t) -> μ_inverted_pendulum(du, u, p, t)
end

function σ_dynamics(cartpole::InvertedPendulum{F}, a::F) where F <: Real
    σ_inverted_pendulum
end

dynamics(c::InvertedPendulum{F}, a::F) where F <: Real = (μ_dynamics(c, a), σ_dynamics(c, a))

state_vec(s::InvertedPendulumState) = [s.x, s.dx, s.θ, s.dθ]
state_vec(c::InvertedPendulum) = state_vec(c.state)

function plot(cartpole::InvertedPendulum)
    l = cartpole.l
    w = PLOT_WIDTH
    h = w * (l + 0.1) / X_THRESH
    plot(xlims=(-X_THRESH, X_THRESH), ylims=(-l -0.1, l + 0.1), legend=false, size=(w, h))#, border=:none)
    s = cartpole.state
    # cart
    cw = 0.5f0 * l
    plot!([s.x - cw / 2, s.x - cw / 2, s.x + cw / 2, s.x + cw / 2], [-0.05, 0, 0, -0.05];
          seriestype=:shape)
    # pole
    plot!([s.x, s.x + l * sin(s.θ)], [0, l * cos(s.θ)]; linewidth=3)
end

