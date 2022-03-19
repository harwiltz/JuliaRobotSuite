export AbstractRobotEnv, LazyValue, BaseMeshCatVisualizer, acquire!

using MeshCat

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
