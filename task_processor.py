# adapted from https://stackoverflow.com/questions/53493973/how-to-pause-processes-in-case-they-are-consuming-too-much-memory
import time
from threading import Thread
from collections import deque
from multiprocessing import active_children
from tqdm import tqdm
import psutil, time




class TaskProcessor(Thread):
    """Processor class which monitors memory usage for running
    tasks (processes). Suspends execution for tasks surpassing
    `max_mib` and completes them one by one, after behaving
    tasks have finished.
    Adapted to suspend process with lowest memory usage if total memory limit is exceeded and
    start suspended process as soon as another process finished.
    """

    #def __init__(self, n_cores, max_gb, tasks):
    def __init__(self, n_cores, min_mem, max_mem, tasks):
        super().__init__()
        self.n_cores = n_cores
        self.actual_n_cores = n_cores
        self.min_mem = min_mem
        self.max_mem = max_mem
        self.active = 0
        #self.max_gb = max_gb
        #self.max_mib = max_gb * 953.67431640625  # memory threshold

        #self.total_tasks = len(tasks)
        #self.finished_tasks = 0
        self.progress = tqdm(total=len(tasks))
       
        self.tasks = deque(tasks)
        self.suspend_time = 0
        self._running_tasks = []
        self._suspended_tasks = []
        #self.mem_use = psutil.virtual_memory().percent
        
    def run(self):
        """Main-function in new thread."""
        self._update_running_tasks()
        self._monitor_running_tasks()
        self._process_suspended_tasks()    # not sure if they can be some left
        #self._process_last_process()
        self.progress.close()
        


    def _update_running_tasks(self):
        """Start new tasks if we have less running tasks than cores."""
     
        #while len(self._running_tasks) < self.n_cores and len(self.tasks) > 0 and psutil.virtual_memory().percent <= 80:
        while len(self._running_tasks) < self.actual_n_cores and len(self.tasks) > 0:
            p = self.tasks.popleft()
            p.start()
            # for further process-management we here just need the
            # psutil.Process wrapper
            self._running_tasks.append(psutil.Process(pid=p.pid))
            print(f'Started process: {self._running_tasks[-1]}')
            self.active += 1
                
        while len(self._running_tasks) < self.n_cores and len(self._suspended_tasks) > 0:
            sp = self._suspended_tasks[0]
            print(f'Resuming process: {sp}')
            sp.resume()
            self._running_tasks.append(sp)
            self._suspended_tasks.remove(sp)
            self.active += 1
            #time.sleep(1)
           


            
            

    def _monitor_running_tasks(self):
        """Monitor running tasks. Replace completed tasks and suspend tasks
        which exceed the memory threshold `self.max_mib`.
        """
        # loop while we have running or non-started tasks
        #resumed_since = 999
        while self._running_tasks or self.tasks or self._suspended_tasks:
            active_children()  # Joins all finished processes.
            # Without it, p.is_running() below on Unix would not return
            # `False` for finished processes.
            
            self._update_running_tasks()
            actual_tasks = self._running_tasks.copy()
            #total_mem = 0
            #active_mems = []
            #run = False
            run_indices = []
            for i, p in enumerate(actual_tasks):
                if not p.is_running():  # process has finished
                    self._running_tasks.remove(p)
                    #print(f'Removed finished process: {p}')
                    self.active -= 1
                    self.progress.update(1)
                else:
                    run_indices.append(i)
                    #run = True
            mem_use = psutil.virtual_memory().percent
            print(mem_use, self.active)
            
            if mem_use > self.max_mem and self.active > 1:
                active_children()
                sp = actual_tasks[run_indices[0]]
                if sp.is_running():
                    sp.suspend()
                    print(f'Suspended process: {sp}')
                    self._running_tasks.remove(sp)
                    self._suspended_tasks.append(sp)
                    self.suspend_time = time.time()
                    self.active -= 1
                    if self.actual_n_cores > 1: self.actual_n_cores -= 1

            if mem_use < self.min_mem and self.actual_n_cores < self.n_cores:
                self.actual_n_cores += 1

                #except:
                #    print(f'suspend error')


            



            #active_children()
            #actual_tasks = self._running_tasks.copy()
            # don;t suspend last running process and hope for the best
            #if len(self._running_tasks) > 1 or (len(self.tasks) + len(self._suspended_tasks)) >= 1:
            #   
            
            
            #mem_use = psutil.virtual_memory().percent 
            #print(mem_use)
            #while mem_use > self.max_mem and len(run_indices) > 1:
            '''
            mem_use = psutil.virtual_memory().percent
            print(mem_use, self.active)

            if mem_use > self.max_mem:
                
                for i, p in enumerate(actual_tasks):
                    if self.active == 1: break
                    #sp = actual_tasks[run_indices[0]]
                    #try:
                    if p.is_running() and mem_use > self.max_mem:
                        print(f'Suspendeding process: {p}')
                        p.suspend()
                        self._running_tasks.remove(p)
                        self._suspended_tasks.append(p)
                        self.active -= 1
                        #time.sleep(1)
                        mem_use = psutil.virtual_memory().percent
                        print(mem_use, self.active)
                        active_children()
                    #if mem_use <= self.max_mem or len(actual_tasks) == 1: break
                    
                    #except: pass
                    #run_indices.remove(run_indices[0])
            '''

            #if run and actual_tasks and psutil.virtual_memory().percent > 80:#self.max_mem:
            #    sp = actual_tasks[run_indices[0]]
            #    print(f'Suspend process: {sp}')
            #    self._running_tasks.remove(sp)
            #    self._suspended_tasks.append(sp)
            #    sp.suspend()
                        
                    



            '''
                    if len(self._running_tasks) < self.n_cores and len(self._suspended_tasks) > 0:  # do until all tasks started
                        resume_p = self._suspended_tasks[0]
                        print(f'Resuming process: {resume_p}')
                        resume_p.resume()
                        self._running_tasks.append(psutil.Process(pid=resume_p.pid))
                        self._suspended_tasks.remove(resume_p)
            ...
                        
            #actual_tasks = self._running_tasks.copy()'''
             #   elif p.is_running():
             #       try:    # ProcessLookupError: [Errno 3] No such process (originated from proc_pidinfo()) # at some point in similarity process, why?
             #           p_mem = p.memory_info().rss / 2 ** 20
             #           total_mem += p_mem
             #           active_mems.append((p_mem, p))
             #       except: print(f'Not found:', p)
            #print(total_mem)
            #mem_use = psutil.virtual_memory().percent
            #print(mem_use)
            #if total_mem > self.max_mib:    # 40GB
            #    min_mem_p = min(active_mems)[1]
            #    min_mem_p.suspend()
            #    self._running_tasks.remove(min_mem_p)
            #    self._suspended_tasks.append(min_mem_p)

            #actual_tasks = self._running_tasks.copy()
            #if mem_use >= 90:
            #    if not self._running_tasks: break       # ?
            #    rp = self._running_tasks[0]
            #    for r in actual_tasks[1:]:
            #        if r.memory_full_info().uss > rp.memory_full_info().uss:
            #            rp = r
            #    self._running_tasks.remove(psutil.Process(pid=rp.pid))
            #    #self._suspended_tasks.append((p.memory_full_info().uss,  psutil.Process(pid=rp.pid)))
            #    self._suspended_tasks.append((rp.memory_percent(memtype='uss'),  psutil.Process(pid=rp.pid)))
            #    rp.suspend()        # suspend process with highest mem usage
                #self._suspended_tasks.sort()       # sorting to resume process with lowest mem usage first
                #print(f'Suspended process: {min_mem_p}')
            #    print(f'Suspended process: {rp}')
                    #if p.memory_info().rss / 2 ** 20 > self.max_mib:
                    #    p.suspend()
                    #    self._running_tasks.remove(p)
                    #    self._suspended_tasks.append(p)
                    #    print(f'Suspended process: {p}')
            #else:

            #    mem_use = psutil.virtual_memory().percent
            #if self._suspended_tasks:
            #    resume_p = min(self._suspended_tasks)
                #mem_percent = min(self._suspended_tasks)[0].memory_percent(memtype='uss')   # memory percentage of suspended task with lowest memory usage
                #if mem_use <= (90-mem_percent):
            #    if mem_use <= (80 - resume_p[0]):
            #        if len(self._running_tasks) < self.n_cores and len(self._suspended_tasks) > 0:  
                        #resume_p = self._suspended_tasks[0] 
                        #resume_p = min(self._suspended_tasks)
            #            print(f'Resuming process: {resume_p[1]}')
            #            resume_p[1].resume()
            #            self._running_tasks.append(psutil.Process(pid=resume_p[1].pid))
            #            self._suspended_tasks.remove(resume_p)
                        #resumed_since = 0

            

            
            
            
            
            # if all suspended, increase to 60GB and halt if it doesn't work :(
            '''
            if len(self._running_tasks) + len(self.tasks) + len(self._suspended_tasks) == 1:
                break
                
            '''
            time.sleep(1)
            #resumed_since += 1
            
            


    def _process_suspended_tasks(self):         
        """Resume processing of suspended tasks."""
        for p in self._suspended_tasks:
            print(f'\nResuming process: {p}')
            p.resume()
            p.wait()

    # def _process_last_process(self):
        # while self._running_tasks:
        #   actual_tasks = self._running_tasks.copy()
        #    for p in actual_tasks:                     # there should only be one task
        #        if not p.is_running():  # process has finished
        #            self._running_tasks.remove(p)
        #            self.progress.update(1)



