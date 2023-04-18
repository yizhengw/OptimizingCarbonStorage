
data = Vector{Float64}
data = range(1.1, 10.0, length=20)


schedule_idx = [1 1 1 2 2 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4]
schedule_idx_v = vec(schedule_idx)


schedule_index_unique = unique!(schedule_idx_v)

for i in 1:size(schedule_index_unique)[1]


    current_idx = schedule_index_unique[i];
    push!(search_idx, current_idx)
    
    println(search_idx)

    # is_member = ismember(schedule.step.control, search_idx);
    # indexies = find(is_member);

end
