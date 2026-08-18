using JuMP
using SCIP

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

function mrs(x) #to validate feasibility of our solution, is equal to 0 if the given graph is a ramsey witness
	penalty = (r_edges - 1) * binomial(n - 2, r - 2) + (s_edges - 1) * binomial(n - 2, s - 2)
	penalty *= 0.6
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

model = Model(SCIP.Optimizer)
@variable(model, x[1:div(n * (n - 1), 2)], Bin)

# r-clique term
comb = collect(1:r)
while true
    @constraint(model, sum(x[coor(i, j)] for i in comb for j in comb if i < j) <= r_edges - 1)
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
    @constraint(model, 1 <= sum(x[coor(i, j)] for i in comb for j in comb if i < j))
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

dir = zeros(div(n * (n - 1), 2))
@objective(model, Max, dir' * x)
optimize!(model)
assert_is_solved_and_feasible(model)
solution_summary(model)
@show(value(x))
@show(mrs(value(x)))
