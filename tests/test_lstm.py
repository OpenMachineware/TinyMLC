from termcolor import cprint

# Define different level log functions
def log_debug(msg):
    cprint(f"[DEBUG] {msg}", "cyan")  # Cyan: for debug info

def log_info(msg):
    cprint(f"[INFO] {msg}", "green")  # Green: for normal runtime info
    cprint(f"[DEFAULT] {msg}", None)

def log_warning(msg):
    cprint(f"[WARNING] {msg}", "yellow")  # Yellow: for warnings

def log_error(msg):
    cprint("\n" + "=" * 60, "red")
    cprint(f"[ERROR] {msg}", "red", attrs=["bold"])  # Red bold: for errors
    cprint("\n" + "=" * 60, "red")


# --- Test the effect ---
log_debug("Connecting to database...")
log_info("User login successful!")
log_warning("Memory usage reached 85%, please note.")
log_error("Cannot read config file, program will exit!")