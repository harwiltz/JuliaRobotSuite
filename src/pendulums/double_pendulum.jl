module DoublePendulumModule

using ..JuliaRobotSuite

using LinearAlgebra
using StaticArrays

import RigidBodyDynamics as RBD

const G = -9.81

mutable struct DoublePendulumEnv <: AbstractRobotEnv
    pendulum::RBD.Mechanism
end

function DoublePendulumEnv(
    ;
    F = Float64,
    str_world = "world",
    I₁ = 0.333,
    c₁ = 0.5,
    m₁ = 1.,
    l₁ = 1.,
    I₂ = 0.333,
    c₂ = 0.5,
    m₂ = 1.,
)
    world = RBD.RigidBody{F}(str_world)
    pendulum = RBD.Mechanism(world; gravity = SVector(0, 0, G))
    axis = SVector(0., 1., 0.)
    frame₁ = RBD.CartesianFrame3D("upperlink")
    inertia₁ = RBD.SpatialInertia(frame₁,
                                  moment = I₁ * axis * axis',
                                  com = SVector(0, 0, -c₁),
                                  mass = m₁)
    upperlink = RBD.RigidBody(inertia₁)
    shoulder = RBD.Joint("shoulder", RBD.Revolute(axis))
    before_shoulder_to_world = one(RBD.Transform3D, RBD.frame_before(shoulder), RBD.default_frame(world))
    RBD.attach!(pendulum, world, upperlink, shoulder, joint_pose = before_shoulder_to_world)

    frame₂ = RBD.CartesianFrame3D("lowerlink")
    inertia₂ = RBD.SpatialInertia(frame₂,
                                  moment = I₂ * axis * axis',
                                  com = SVector(0, 0, -c₂),
                                  mass = m₂)
    lowerlink = RBD.RigidBody(inertia₂)
    elbow = RBD.Joint("elbow", RBD.Revolute(axis))
    before_elbow_to_after_shoulder = RBD.Transform3D(RBD.frame_before(elbow),
                                                     RBD.frame_after(shoulder),
                                                     SVector(0, 0, -l₁))
    RBD.attach!(pendulum, upperlink, lowerlink, elbow, joint_pose = before_elbow_to_after_shoulder)

    state = RBD.MechanismState(pendulum)
    RBD.set_configuration!(state, shoulder, 0.3)
    RBD.set_configuration!(state, elbow, 0.4)
    RBD.set_velocity!(state, shoulder, 1.)
    RBD.set_velocity!(state, elbow, 2.)

    DoublePendulumEnv(pendulum)
end

end
