from typing import List, Dict
import json
import datetime

class LogAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def get_errors_by_period(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        errors = []
        with open(self.log_file) as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    log_time = datetime.fromisoformat(log_entry['timestamp'])
                    
                    if (log_time >= start_time and 
                        log_time <= end_time and 
                        log_entry['level'] in ['ERROR', 'CRITICAL']):
                        errors.append(log_entry)
                except:
                    continue
        return errors