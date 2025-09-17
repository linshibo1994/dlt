
import subprocess
import json
from typing import List, Dict, Any

def detect_gpu_nvidia_smi() -> List[Dict[str, Any]]:
    """Detect GPU using nvidia-smi directly"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            gpus = []
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpus.append({
                            'index': i,
                            'name': parts[0],
                            'memory_total_mb': int(parts[1]),
                            'memory_free_mb': int(parts[2]),
                            'memory_used_mb': int(parts[3]),
                            'utilization_percent': int(parts[4]),
                            'vendor': 'NVIDIA'
                        })
            return gpus
    except Exception as e:
        print(f"GPU detection error: {e}")

    return []

def get_gpu_acceleration_available():
    """Check if GPU acceleration is available"""
    gpus = detect_gpu_nvidia_smi()
    return len(gpus) > 0, gpus

if __name__ == "__main__":
    available, gpus = get_gpu_acceleration_available()
    print(f"GPU acceleration available: {available}")
    if available:
        for gpu in gpus:
            print(f"  {gpu['name']}: {gpu['memory_total_mb']} MB")
