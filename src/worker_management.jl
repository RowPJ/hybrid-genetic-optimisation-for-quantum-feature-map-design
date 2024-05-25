# NOTE: this file should only be loaded by the main process since it sets environment variables
# that should not be overwritten in worker processes. loading this file will start up worker
# processes.

using Distributed

include("workers.jl")

"Add workers from a remote machine through SSH with tunneling.
The login details of the machine spec selected need to be
configured in ~/.ssh/config in order for a connection to succeed."
function add_remote_workers(;n_cpu=0, n_gpu=0, machine_spec::String="") #example machine spec: "Rowan@127.0.0.1:22"
    if n_cpu < 0 || n_gpu < 0
        error("One of the arguments was negative."*
              "No operation was performed.")
    elseif n_cpu == 0 && n_gpu == 0
        error("Tried to create 0 cpu workers and"*
              " 0 gpu workers. Is this intentional?"*
              " No operation was performed")
    end
    new_ids = []
    # make cpu workers with cpu environment variable
    (n_cpu != 0) && append!(new_ids, addprocs([(machine_spec, n_cpu)]; env=[REGISTER_VARIABLE=>CPU_REGISTER_TYPE], shell=:wincmd, tunnel=true))
    # make gpu workers with gpu environment variable
    (n_gpu != 0) && append!(new_ids, addprocs([(machine_spec, n_gpu)]; env=[REGISTER_VARIABLE=>GPU_REGISTER_TYPE], shell=:wincmd, tunnel=true, sshflags="-vvv"))
    # have new workers load workers.jl
    @everywhere new_ids include("workers.jl")
    return nothing
end

"Adds worker processes on the local machine.
Does nothing if count <= 0."
function add_local_cpu_workers(count)
    # ensure count >= 1
    count <= 0 && return nothing
    # create workers
    new_ids = addprocs(count)
    # have new workers load workers.jl
    @everywhere new_ids include("workers.jl")
    return nothing
end

# ensure the main process (and local threads if threading is used) know
# what to use to simulate kernels in the case of there being no worker
# processes (or threading being used)
ENV[REGISTER_VARIABLE] = CPU_REGISTER_TYPE

local_workers_count = parse(Int64, ENV["LOCAL_WORKER_COUNT"])
remote_workers_count = 0

# create workers
target_worker_counts = (local_workers_count, remote_workers_count)
add_local_cpu_workers(target_worker_counts[1] - nprocs() + 1) # +1 since nprocs() includes the main process

# create remote gpu workers on local machine
# add_remote_workers(n_gpu=target_worker_counts[2], machine_spec="rowan@127.0.0.1")

"Convenience function for interactive development that reloads workers.jl on all worker processes.
It should be used when the file is modified with the REPL running to have workers reflect the changes."
function reload_workers_file()
    @everywhere include("workers.jl")
end
