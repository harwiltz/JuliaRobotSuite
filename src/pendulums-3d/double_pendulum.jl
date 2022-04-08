module DoublePendulumModule

using ..JuliaRobotSuite

using LinearAlgebra
using StaticArrays

using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics

import Plots

const PENDULUM_URDF = "models/double-pendulum.urdf"

mutable struct DoublePendulumEnv <: AbstractRobotEnv
    pendulum::Mechanism
    vis::LazyValue{MechanismVisualizer}
end

DoublePendulumEnv() = let
    pendulum = parse_urdf(PENDULUM_URDF)
    init_fn = BaseMechanismVisualizer(pendulum, PENDULUM_URDF)
    DoublePendulumEnv(
        pendulum,
        LazyValue{MechanismVisualizer}(init_fn)
    )
end

function Plots.plot(env::DoublePendulumEnv)
    vis = acquire!(env.vis)
end

function Base.show(io::IOContext{Base.TTY}, t::MIME{Symbol("text/plain")}, env::DoublePendulumEnv)
    Base.show(env.pendulum)
end

end
