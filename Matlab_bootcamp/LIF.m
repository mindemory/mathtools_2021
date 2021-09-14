% This code will simualte a simple neuron
% The equations here are called the Leaky-Integrate and Fire (LIF) model

T = 10; % total simulation time
dt = 0.1; % simulation timestep
Timesteps = round(T/dt);
v=zeros(1,Timesteps);

count =0;
ref_p = 0;
for j=1:1:Timesteps
    tau =1; % timeconstant of the cell
    i=1.5; % inputs to the cell
    if ref_p == 0
        dv = (-v(j) + i)/tau; % compute voltage update
        v(j+1) =v(j)+ dv*dt; % update the next voltage value
    else
        ref_p = ref_p - 1;
       % v(j+1) = 0;
    end
    
    % Write code to check if neuron spiked (when v>=1)
    % if the neuron spiked, set the voltage to zero (this is called reset)
    % then, keep a count of how many times this neuron spiked in time T
    
    if v(j+1) >= 1
        ref_p = 5;
        count = count+1;
        v(j+1) = 0;
    end
    
    % for a more advanced exercise, set a refractory period:
    % a neuron stays at zero voltage for 5 timesteps after spiking
    
    %ref_p = ref_p + 5;
end

figure;
plot(v);
ylabel('Voltage');
xlabel('Time');
title('My neuron');