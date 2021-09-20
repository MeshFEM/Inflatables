import re, os, sys, subprocess

class ParameterSweepJobs:
    def __init__(self, config):
        self.config = config
        self.parameterChoices = config.parameterChoices
        self.parameterNames = list(self.parameterChoices.keys())
        self.outputRootDirectory = config.outputRootDirectory
        self.parameterShortNames = config.parameterShortNames
        self.parameterFormatters = config.parameterFormatters

    def numJobs(self):
        import numpy as np
        return np.product([len(c) for p, c in self.parameterChoices.items()])

    def paramsWithChoices(self):
        return [n for n, v in self.config.parameterChoices.items() if len(v) > 1]

    def choicelessParams(self):
        return [p for p in self.parameterNames if p not in self.paramsWithChoices()]

    def __len__(self):
        return self.numJobs()

    def numParameters(self):
        return len(self.parameterNames)

    def validIndex(self, i):
        return i in range(0, self.numJobs())

    def parametersForJob(self, jnum):
        import utils
        if (not self.validIndex(jnum)): raise Exception(f'Job index {jnum} out of bounds')
        return {name: value for name, value in zip(self.parameterNames,
                                                   utils.nth_choice(jnum, *[self.parameterChoices[key] for key in self.parameterNames]))}
    def parameterName(self, i, useShortName = True):
        """Get the parameter's short name, defaulting to its full name if one wasn't specified."""
        if (not useShortName): return self.parameterNames[i]
        return self.parameterShortNames.get(self.parameterNames[i], self.parameterNames[i])

    def jobName(self, jnum, param_sep = '/', value_sep='_', ignoreParams = [], useShortName = True):
        params = self.parametersForJob(jnum)
        def parameterString(i):
            param = self.parameterNames[i]
            if param in ignoreParams: return ''
            name = self.parameterName(i, useShortName)
            value = params[param]
            formattedVal = self.parameterFormatters.get(param, lambda x: x)(params[param])
            if name == '': return formattedVal
            return f'{name}{value_sep}{formattedVal}'
        paramStrings = [parameterString(i) for i in range(self.numParameters())]
        return param_sep.join([s for s in paramStrings if s])

    def directoryForJob(self, jnum, param_sep = '/', value_sep='_'):
        subdir = self.jobName(jnum, param_sep, value_sep)
        return f'{self.outputRootDirectory}/{subdir}'

    def cameraParamsForJob(self, jnum):
        params = self.parametersForJob(jnum)
        name = params['input']['name']
        try:
            for k in self.config.cameraParamsForInput:
                if re.match(k, name) is not None:
                    return self.config.cameraParamsForInput[k]
        except: pass
        return None

# Number of jobs to pack within a single task invocation (using a thread pool).
# This is necessary on HPC1 since its Slurm installation is unable
# to schedule jobs on fewer than 5 cpus.
CHUNK_SIZE = 5

def batchfile(jobs, script, jobIndices, workingDir=None, cpusPerTask=CHUNK_SIZE, memPerTask='30gb', time='24:00:00'):
    """
    A Slurm batch file for a single task (no longer an array job so that we can
    name the task with a list of the chunked jobs within it).
    """
    if workingDir is None:
        workingDir = os.path.dirname(os.path.realpath(__file__))
    import numpy as np

    jobString = ','.join(map(str, jobIndices))

    import inspect
    return inspect.cleandoc(f"""
        #!/bin/bash
        #SBATCH --job-name={jobs.config.name}_{jobString}
        #SBATCH --chdir {workingDir}
        #SBATCH --nodes 1
        #SBATCH --partition=low
        #SBATCH --cpus-per-task {cpusPerTask}
        #SBATCH --ntasks 1
        #SBATCH --mem {memPerTask}
        #SBATCH --time {time}
        #SBATCH --output=/dev/null
        #SBATCH --error=/dev/null

        . ~/environment.sh

        python {script} {jobString}
        """)

def parseJobString(jstr):
    """
    Parse a Slurm-style job string in the form "1-3,5,8,10-11"
    """
    def parseInterval(interval):
        i = [int(j) for j in interval.split('-')]
        if len(i) == 1: return i
        if len(i) == 2: return range(i[0], i[1] + 1)
        raise Exception('Invalid task interval ' + interval)
    jstr = jstr.translate(str.maketrans({' ': '', '\t': ''})) # remove whitespace
    return list({j for i in jstr.split(',') for j in parseInterval(i)}) # list of unique jobs

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # From https://docs.python.org/3/library/itertools.html
    import itertools
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def submit_jobs(jobs, script, jobIndices=None, alreadyGrouped=False):
    # Divide requested jobs into chunked tasks (comma-separated job indices) of approximately CHUNK_SIZE jobs each
    if jobIndices is None: jobIndices = list(range(len(jobs)))

    if not alreadyGrouped:
        tasks = grouper(jobIndices, CHUNK_SIZE)
    else:
        tasks = jobIndices

    for taskJobs in tasks:
        taskJobs = [j for j in taskJobs if j is not None] # `grouper` pads out the last chunk with None; remove them
        batchfile_string = batchfile(jobs, script, taskJobs)
        print(batchfile_string)

        p = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE)
        p.communicate(input=(batchfile_string + '\n').encode('utf-8'))

def is_resumable(jobs, jobIdx, check_func):
    status, resumable = check_func(jobs, jobIdx)
    return resumable

def resumable_jobs(jobs, check_func):
    resumable = [i for i in range(len(jobs)) if is_resumable(jobs, i, check_func)]
    return ','.join(map(str, resumable))

def usage(mainScriptName):
    print(f'Usage: python {mainScriptName} my_config.py [list,info,status,run,resume,analyze,submit,submit_resume,submit_analyze]')
    sys.exit(-1)

def do_cli(run_func, analysis_func, check_func, resume_func):
    mainScriptName = sys.modules['__main__'].__file__

    if (len(sys.argv) < 3): usage(mainScriptName)
    configFile = sys.argv[1]
    action = sys.argv[2]

    import importlib
    spec = importlib.util.spec_from_file_location('config', configFile)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Allow the config file to override the default threads count used for
    # CHOLMOD/MeshFEM
    if hasattr(config, 'num_threads'):
        if ('numpy' in sys.modules) or ('MeshFEM' in sys.modules):
            print('********************************************************************************')
            print('WARNING: `num_threads` cannot take full effect if numpy/MeshFEM/matplotlib/etc.\nare imported before `do_cli` is called (or from the job config file)!')
            print('********************************************************************************')
        nthreads = config.num_threads
        for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'LP_NUM_THREADS', # number of LLVMpipe threads for OSMesa rendering
                    'MESHFEM_NUM_THREADS']:
            os.environ[var] = str(nthreads)
        import MeshFEM, parallelism
        parallelism.set_max_num_tbb_threads(nthreads)

    if (action.lower() == 'debug') or (action.lower() == 'debug_resume'):
        config.redirect_io = False
        config.outputRootDirectory = 'debug'
        config.debug = True
        action = 'run' if (action.lower() == 'debug') else 'resume'

    jobs = ParameterSweepJobs(config)

    if (action.lower() == 'list'):
        for i in range(jobs.numJobs()):
            print(f'{i}:\t', jobs.jobName(i))

    elif (action.lower() == 'info'):
        job_idx, = sys.argv[3:]
        job_idx = int(job_idx)
        print(f'result directory: {jobs.directoryForJob(job_idx)}')
        print(f'parameters: {jobs.parametersForJob(job_idx)}')

    elif (action.lower() == 'analyze'):
        # Analyze a sequence of jobs one after the other
        analysisDir, jobString = sys.argv[3:]
        for job_idx in parseJobString(jobString):
            analysis_func(jobs, int(job_idx), analysisDir)

    elif (action.lower() == 'analyze_all'):
        analysisDir, = sys.argv[3:]
        for i in range(len(jobs)):
            analysis_func(jobs, i, analysisDir)

    elif (action.lower() == 'status' or action.lower() == 'check'):
        statusName = lambda i: '.'.join(str(check_func(jobs, i)[0]).split('.')[1:])
        statuses = [statusName(i) for i in range(len(jobs))]
        statusWidth = max(map(len, statuses)) + 1
        for i in range(len(jobs)):
            print(f'{i:<4} {statuses[i]:<{statusWidth}} {jobs.directoryForJob(i)}')

    # Run a single-thread worker or worker pool
    elif (action.lower() == 'run') or (action.lower() == 'resume'):
        resuming = (action.lower() == 'resume')
        task_func = resume_func if resuming else run_func
        args = sys.argv[3:]
        badArgs = Exception(f'Invalid arguments for `{action}`: should be either a single job id or a job list to run in parallel, with an optional second arguemnt specifying thel number of worker threads (default CHUNK_SIZE)')
        if len(args) == 1: jobString, threadpoolSize = args[0], CHUNK_SIZE
        elif len(args) == 2: jobString, threadpoolSize = args[0]
        else: raise badArgs

        job_idxs = parseJobString(jobString)
        if len(job_idxs) == 1: # a single job to run directly in this process
            task_func(jobs, job_idxs[0])
        elif len(job_idxs) > 1:
            # Run multiple jobs in parallel in a thread pool of size CHUNK_SIZE
            from multiprocessing.pool import ThreadPool
            def work(ji):
                print(('resuming' if resuming else 'running') + f' job {ji}')
                ret = subprocess.call(['python', mainScriptName, configFile, action, str(ji)], env=os.environ, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                print(f'job {ji} exited with status {ret}')
                return ret
            p = ThreadPool(threadpoolSize)
            p.map(work, job_idxs)
        else: raise badArgs

    elif (action.lower() == 'resume'):
        job_idx, = sys.argv[3:]
        resume_func(jobs, int(job_idx))

    elif action.lower() == 'submit':
        jobIndicesString, jobIndices = None, None
        try: jobIndicesString, = sys.argv[3:]
        except: pass
        if jobIndicesString is not None: jobIndices = parseJobString(jobIndicesString)
        submit_jobs(jobs, f'{mainScriptName} {configFile} run', jobIndices=jobIndices)

    elif action.lower() == 'submit_resume':
        try: jobIndicesString, = sys.argv[3:]
        except:
            jobIndicesString = resumable_jobs(jobs, check_func)
            print(f'Resuming jobs: {jobIndicesString}')
        jobIndices = [j for j in parseJobString(jobIndicesString) if is_resumable(jobs, j, check_func)]
        if (len(jobIndices) == 0):
            print("Nothing to resume")
            return
        submit_jobs(jobs, f'{mainScriptName} {configFile} resume', jobIndices=jobIndices)

    elif action.lower() == 'submit_analyze':
        args = sys.argv[3:]
        if len(args) == 2:
            analysisDir, jobString, nslurmjobs = args[0], args[1], '1'
        if len(args) == 3:
            analysisDir, jobString, nslurmjobs = args[0:]
        else:
            print(f'usage: python {mainScriptName} my_config.py submit_analyze analysis_dir job_index_string [nslurmjobs=1]')
            sys.exit(-1)

        # Split list of jobs to analyze into `nslurmjobs` sub-lists of roughly
        # equal length to be processed by separate analysis batch jobs.
        jidxs = parseJobString(jobString)
        import numpy as np
        tasks = np.array_split(jidxs, int(nslurmjobs))
        submit_jobs(jobs, f'{mainScriptName} {configFile} analyze {analysisDir}', jobIndices=tasks, alreadyGrouped=True)

    else:
        usage(mainScriptName)
        raise Exception(f'Unknown action `{action}`')
