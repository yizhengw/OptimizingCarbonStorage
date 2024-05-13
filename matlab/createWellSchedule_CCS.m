
function [schedule] = createWellSchedule_CCS(TOTAL_TIME, INJECTION_STOP, NSTEP_INJECTION, ...
    NSTEP_SHUT_IN, W, INTERVAL, varargin)


p = inputParser;
addRequired(p, 'TOTAL_TIME');
addRequired(p, 'INJECTION_STOP');
addRequired(p, 'NSTEP_INJECTION');
addRequired(p, 'NSTEP_SHUT_IN');
addRequired(p, 'W');
addRequired(p, 'INTERVAL');
addParameter(p, 'bc', []);

parse(p, TOTAL_TIME, INJECTION_STOP, NSTEP_INJECTION, ...
    NSTEP_SHUT_IN, W, INTERVAL, varargin{:});

interval = INTERVAL * year;

simTime = TOTAL_TIME*year;
simTime_injection = INJECTION_STOP*year;
simTime_shut_in = simTime - simTime_injection;

%assert(rem(INJECTION_STOP/NSTEP_INJECTION,INTERVAL)==0,'The well intervall resolution does not fit the timestepping');


timesteps = [repmat(simTime_injection/NSTEP_INJECTION, NSTEP_INJECTION, 1); ...
    repmat(simTime_shut_in/NSTEP_SHUT_IN, NSTEP_SHUT_IN, 1)];

% schedule function:

n_wells = length(W);
n_steps = length(timesteps);
cumtime = cumsum(timesteps);

schedule = struct();
schedule.step.val = timesteps;

schedule.step.control = zeros(size(timesteps));

for i = 1:n_wells
    step_wells = W;
    for j = (i+1):n_wells
        step_wells(j).type = 'rate';
        step_wells(j).val = 0;
    end
    schedule.control(i).W = step_wells;
    schedule.control(i).bc = p.Results.bc;

    time_idxs = cumtime > interval*(i-1);
    schedule.step.control(time_idxs) = i;

end



end
