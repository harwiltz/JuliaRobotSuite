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
    vis::LazyValue{Visualizer}
end

DoublePendulumEnv() = DoublePendulumEnv(parse_urdf(PENDULUM_URDF),
                                        LazyValue{Visualizer}(BaseMeshCatVisualizer))

function Plots.plot(env::DoublePendulumEnv)
    vis = acquire!(env.vis)
    mvis = MechanismVisualizer(env.pendulum, URDFVisuals(PENDULUM_URDF), vis)
end

function Base.show(io::IOContext{Base.TTY}, t::MIME{Symbol("text/plain")}, env::DoublePendulumEnv)
    Base.show(env.pendulum)
end

end
