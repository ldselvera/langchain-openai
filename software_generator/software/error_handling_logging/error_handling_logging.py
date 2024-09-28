class ErrorHandlingLogging:
    def __init__(self):
        pass
    
    def handle_errors(self, error_message):
        """
        Handles errors gracefully and provides informative error messages to users.
        
        Args:
        - error_message: str, the error message to be displayed
        
        Returns:
        - None
        """
        print("Error occurred:", error_message)
    
    def log_progress(self, progress_message):
        """
        Logs the progress and results of each AutoML run.
        
        Args:
        - progress_message: str, the progress message to be logged
        
        Returns:
        - None
        """
        print("Progress:", progress_message)