# adapted from https://stackoverflow.com/questions/53493973/how-to-pause-processes-in-case-they-are-consuming-too-much-memory
import time
from threading import Thread
from collections import deque
from multiprocessing import active_children
from tqdm import tqdm
import psutil


class TaskProcessor(Thread):
    """Processor class which monitors memory usage for running
    tasks (processes). Suspends execution for tasks surpassing
    `max_mib` and completes them one by one, after behaving
    tasks have finished.
    Adapted to suspend process with lowest memory usage if total memory limit is exceeded and
    start suspended process as soon as another process finished.
    """

    def __init__(self, n_cores, max_gb, tasks):
        super().__init__()
        self.n_cores = n_cores
        self.max_mib = max_gb * 953.67431640625  # memory threshold

        #self.total_tasks = len(tasks)
        #self.finished_tasks = 0
        self.progress = tqdm(total=len(tasks))
        self.tasks = deque(tasks)

        self._running_tasks = []
        self._suspended_tasks = []
        

    def run(self):
        """Main-function in new thread."""
        self._update_running_tasks()
        self._monitor_running_tasks()
        #self._process_suspended_tasks()    #only last suspended
        self.progress.close()

    def _update_running_tasks(self):
        """Start new tasks if we have less running tasks than cores."""
        while len(self._running_tasks) < self.n_cores and len(self.tasks) > 0:
            p = self.tasks.popleft()
            p.start()
            # for further process-management we here just need the
            # psutil.Process wrapper
            self._running_tasks.append(psutil.Process(pid=p.pid))
            print(f'Started process: {self._running_tasks[-1]}')
            

    def _monitor_running_tasks(self):
        """Monitor running tasks. Replace completed tasks and suspend tasks
        which exceed the memory threshold `self.max_mib`.
        """
        # loop while we have running or non-started tasks
        while self._running_tasks or self.tasks:
            active_children()  # Joins all finished processes.
            # Without it, p.is_running() below on Unix would not return
            # `False` for finished processes.
            self._update_running_tasks()
            actual_tasks = self._running_tasks.copy()
            total_mem = 0
            active_mems = []
            for p in actual_tasks:
                if not p.is_running():  # process has finished
                    self._running_tasks.remove(p)
                    print(f'Removed finished process: {p}')
                    #self.finished_tasks += 1
                    self.progress.update(1)
                    if len(self._running_tasks) < self.n_cores and len(self._suspended_tasks) > 0:  # do until all tasks started
                        resume_p = self._suspended_tasks[0]
                        print(f'\nResuming process: {resume_p}')
                        resume_p.resume()
                        self._running_tasks.append(psutil.Process(pid=resume_p.pid))
                        self._suspended_tasks.remove(resume_p)
                        actual_tasks = self._running_tasks.copy()


            for p in actual_tasks:
                if p.is_running():
                    p_mem = p.memory_info().rss / 2 ** 20
                    total_mem += p_mem
                    active_mems.append((p_mem, p))
            
            if total_mem > self.max_mib:    # 40GB
                min_mem_p = min(active_mems)[1]
                min_mem_p.suspend()
                self._running_tasks.remove(min_mem_p)
                self._suspended_tasks.append(min_mem_p)
                print(f'Suspended process: {min_mem_p}')
                    #if p.memory_info().rss / 2 ** 20 > self.max_mib:
                    #    p.suspend()
                    #    self._running_tasks.remove(p)
                    #    self._suspended_tasks.append(p)
                    #    print(f'Suspended process: {p}')
            
            #if self.finished_tasks == self.total_tasks - 1: # if all but one finished, break and process last suspended track
            #    break

            time.sleep(1)


'''
    def _process_suspended_tasks(self):
        """Resume processing of suspended tasks."""
        for p in self._suspended_tasks:
            print(f'\nResuming process: {p}')
            p.resume()
            p.wait()
'''


