export AbstractRobotEnv, LazyValue, BaseMeshCatVisualizer, BaseMechanismVisualizer, acquire!

using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms

abstract type AbstractRobotEnv <: AbstractEnv end

mutable struct LazyValue{V}
    init::Any
    val::Union{Missing, V}
end

LazyValue{V}(init) where {V} = LazyValue{V}(init, missing)

function acquire!(lv::LazyValue)
    if ismissing(lv.val)
        setfield!(lv, :val, lv.init())
    end
    lv.val
end

function BaseMeshCatVisualizer(window = true)
    vis = Visualizer()
    if window
        open(vis)
    end
    vis
end

function BaseMechanismVisualizer(mechanism::Mechanism, urdf::String; window = true)
    () -> begin
        vis = Visualizer()
        if window
            open(vis)
        end
        MechanismVisualizer(mechanism, URDFVisuals(urdf), vis)
    end
end
