import os
import sys
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*pynvml.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress pynvml and pkg_resources warnings specifically
import logging as _logging
_logging.getLogger('torch.cuda').setLevel(_logging.ERROR)
_logging.getLogger('mlflow.utils').setLevel(_logging.ERROR)

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Configure file handler with UTF-8 encoding for Windows compatibility
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Configure console handler with UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger("cnnClassifierLogger")