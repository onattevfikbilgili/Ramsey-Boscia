import FrankWolfe
import Boscia
using SCIP
using MathOptInterface
using Bonobo
using LinearAlgebra
using SparseArrays
const MOI = MathOptInterface

n = parse(Int, ARGS[1])
r = parse(Int, ARGS[2])
s = parse(Int, ARGS[3])

const r_edges = r * (r - 1) ÷ 2
const s_edges = s * (s - 1) ÷ 2

function coor(i,j) #tells the coordinate of the entry corresponding to the edge between nodes i and j
    i == j && error("No diagonal entries")
    if i > j
        i, j = j, i
    end
    offset = (i - 1) * n - ((i - 1) * i) ÷ 2
    return offset + (j - i)
end

penalty = (r_edges - 1) * binomial(n - 2, r - 2) + (s_edges - 1) * binomial(n - 2, s - 2)
penalty *= 0.6

function mrs(x)
    qr = r * (r - 1) ÷ 2
    total = 0.0
    # r-clique term
    comb = collect(1:r)
    while true
        product = 1.0
        for i in 1:r-1
            for j in i+1:r
                product *= x[coor(comb[i], comb[j])]
            end
        end
        total += product
        i = r
        while i >= 1 && comb[i] == n - r + i
            i -= 1
        end
        i == 0 && break
        comb[i] += 1
        for j in i+1:r
            comb[j] = comb[j-1] + 1
        end
    end
    # s-independent-set term
    comb = collect(1:s)
    while true
        product = 1.0
        for i in 1:s-1
            for j in i+1:s
                product *= 1 - x[coor(comb[i], comb[j])]
            end
        end
        total += product
        i = s
        while i >= 1 && comb[i] == n - s + i
            i -= 1
        end
        i == 0 && break
        comb[i] += 1
        for j in i+1:s
            comb[j] = comb[j-1] + 1
        end
    end
    #convexification penalty term
    for entry in x
        total += (entry^2 - entry) * penalty
    end
    return total
end

function grad_mrs!(storage, x)
    fill!(storage, 0.0)
    # ----- r-clique term -----
    comb = collect(1:r)
    while true
        edges = Int[]
        for i in 1:r-1
            for j in i+1:r
                push!(edges, coor(comb[i], comb[j]))
            end
        end
        m = length(edges)
        for k in 1:m
            prod_other = 1.0
            for l in 1:m
                l == k && continue
                prod_other *= x[edges[l]]
            end
            storage[edges[k]] += prod_other
        end
        i = r
        while i >= 1 && comb[i] == n - r + i
            i -= 1
        end
        i == 0 && break
        comb[i] += 1
        for j in i+1:r
            comb[j] = comb[j-1] + 1
        end
    end
    # ----- s-independent-set term -----
    comb = collect(1:s)
    while true
        edges = Int[]
        for i in 1:s-1
            for j in i+1:s
                push!(edges, coor(comb[i], comb[j]))
            end
        end
        m = length(edges)
        for k in 1:m
            prod_other = 1.0
            for l in 1:m
                l == k && continue
                prod_other *= (1 - x[edges[l]])
            end
            storage[edges[k]] -= prod_other
        end
        i = s
        while i >= 1 && comb[i] == n - s + i
            i -= 1
        end
        i == 0 && break
        comb[i] += 1
        for j in i+1:s
            comb[j] = comb[j-1] + 1
        end
    end
    # ----- convexification penalty -----
    @inbounds for i in eachindex(x)
        storage[i] += penalty * (2x[i] - 1)
    end
end

function lipschitz_mrs(n, r, s, penalty)
    qr = binomial(r, 2)
    qs = binomial(s, 2)

    return (qr - 1) * binomial(n - 2, r - 2) +
           (qs - 1) * binomial(n - 2, s - 2) +
           2 * penalty
end

L = lipschitz_mrs(n, r, s, penalty)

N = n * (n - 1) ÷ 2
base_lmo = FrankWolfe.BoxLMO(zeros(Float64, N), ones(Float64, N))
lmo = Boscia.ManagedLMO(base_lmo, zeros(Float64, N), ones(Float64, N), collect(1:N), N)
function build_branch_callback()
    return function (tree, node, vidx::Int)
        x = Bonobo.get_relaxed_values(tree, node)
        primal = tree.root.problem.f(x)
        lower_bound = primal - node.dual_gap
        if lower_bound > 0.0 + eps()
            println("No need to branch here. Node lower bound already positive.")
        end
        valid_lower = lower_bound > 0.0 + eps()
        return valid_lower, valid_lower
    end
end
function build_tree_callback()
    return function (tree, node; worse_than_incumbent=false, node_infeasible=false, lb_update=false)
        if isapprox(tree.incumbent, 0.0, atol=eps())
            tree.root.problem.solving_stage = Boscia.USER_STOP
            println("Optimal solution found.")
        end
        if Boscia.tree_lb(tree::Bonobo.BnBTree) > 0.0 + eps()
            tree.root.problem.solving_stage = Boscia.USER_STOP
            println("Tree lower bound already positive. No solution possible.")
        end
    end
end

settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:print_iter] = 10
settings.branch_and_bound[:bnb_callback] = build_tree_callback()
settings.branch_and_bound[:branch_callback] = build_branch_callback()
settings.frank_wolfe[:variant] = Boscia.BlendedDecompositionInvariantConditionalGradient()
settings.frank_wolfe[:line_search] = FrankWolfe.Shortstep(L)
settings.frank_wolfe[:lazy] = true
settings.frank_wolfe[:max_fw_iter] = 1000

x, _, result = Boscia.solve(mrs, grad_mrs!, lmo, settings=settings)

@show(x)
@show(mrs(x))
