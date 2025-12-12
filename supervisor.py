import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('supervisor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

PROGRAMS = [
    {
        'name': 'main.py',
        'path': 'main.py',
        'restart_delay': 5,
        'max_restart_attempts': None,
        'enabled': True
    },
    {
        'name': 'v2.py',
        'path': 'v2.py',
        'restart_delay': 5,
        'max_restart_attempts': None,
        'enabled': False
    }
]

class ProcessMonitor:
    def __init__(self, config):
        self.name = config['name']
        self.path = config['path']
        self.restart_delay = config.get('restart_delay', 5)
        self.max_restart_attempts = config.get('max_restart_attempts', None)
        self.enabled = config.get('enabled', True)
        self.process = None
        self.restart_count = 0
        self.last_start_time = None
        
    def start(self):
        if not self.enabled:
            logger.info(f"{self.name} is disabled, skipping...")
            return False
            
        try:
            logger.info(f"Starting {self.name}...")
            self.process = subprocess.Popen(
                [sys.executable, self.path],
                stdout=None,
                stderr=None,
                stdin=None
            )
            self.last_start_time = datetime.now()
            self.restart_count += 1
            logger.info(f"{self.name} started with PID {self.process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start {self.name}: {e}")
            return False
    
    def is_running(self):
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def get_exit_code(self):
        if self.process:
            return self.process.poll()
        return None
    
    def should_restart(self):
        if not self.enabled:
            return False
        if self.max_restart_attempts is not None and self.restart_count >= self.max_restart_attempts:
            logger.warning(f"{self.name} has reached maximum restart attempts ({self.max_restart_attempts})")
            return False
        return True
    
    def stop(self):
        if self.process and self.is_running():
            logger.info(f"Stopping {self.name}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"{self.name} did not stop gracefully, killing...")
                self.process.kill()

class Supervisor:
    def __init__(self, programs):
        self.monitors = [ProcessMonitor(config) for config in programs]
        self.running = True
        
    def start_all(self):
        logger.info("Supervisor starting all enabled programs...")
        for monitor in self.monitors:
            if monitor.enabled:
                monitor.start()
                time.sleep(1)
    
    def check_and_restart(self):
        for monitor in self.monitors:
            if not monitor.enabled:
                continue
                
            if not monitor.is_running():
                exit_code = monitor.get_exit_code()
                if exit_code is not None:
                    logger.warning(f"{monitor.name} exited with code {exit_code}")
                else:
                    logger.warning(f"{monitor.name} is not running")
                
                if monitor.should_restart():
                    logger.info(f"Restarting {monitor.name} in {monitor.restart_delay} seconds...")
                    time.sleep(monitor.restart_delay)
                    monitor.start()
                else:
                    logger.error(f"{monitor.name} will not be restarted")
    
    def stop_all(self):
        logger.info("Supervisor stopping all programs...")
        for monitor in self.monitors:
            monitor.stop()
        self.running = False
    
    def run(self):
        self.start_all()
        
        try:
            while self.running:
                self.check_and_restart()
                time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Supervisor received shutdown signal")
            self.stop_all()
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            self.stop_all()

def main():
    logger.info("="*60)
    logger.info("Project Gabriel Supervisor starting...")
    logger.info("="*60)
    
    supervisor = Supervisor(PROGRAMS)
    supervisor.run()
    
    logger.info("Supervisor shutdown complete")

if __name__ == "__main__":
    main()
