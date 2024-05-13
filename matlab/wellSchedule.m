function schedule = wellSchedule(timesteps, wells, interval, varargin)
%wellSchedule Create a simulation schedule that adds wells sequentially
%   Detailed explanation goes here

p = inputParser;
addRequired(p, 'timesteps');
addRequired(p, 'wells');
addRequired(p, 'interval');
addParameter(p, 'bc', []);
addParameter(p, 'src', []);
parse(p, timesteps, wells, interval, varargin{:});

n_wells = length(wells);
n_steps = length(timesteps);
cumtime = cumsum(timesteps);

schedule = struct();
schedule.step.val = timesteps;
schedule.step.control = zeros(size(timesteps));
for i = 1:n_wells
    step_wells = wells;
    for j = (i+1):n_wells
        step_wells(j).type = 'rate';
        step_wells(j).val = 0;
    end
    schedule.control(i).W = step_wells; 
    schedule.control(i).src = p.Results.src;
    schedule.control(i).bc = p.Results.bc;
    
    time_idxs = cumtime >= interval*(i-1);
    schedule.step.control(time_idxs) = i;
end
end

