from dataset_handling import DatasetHandling
from architecture_search import ArchitectureSearch
from hyperparameter_search import HyperparameterSearch
from evaluation_metrics import EvaluationMetrics
from model_training_evaluation import ModelTrainingEvaluation
from results_visualization import ResultsVisualization
from exporting_deploying_models import ExportingDeployingModels
from documentation_support import DocumentationSupport
from integration_extensibility import IntegrationExtensibility
from error_handling_logging import ErrorHandlingLogging
from performance_optimization import PerformanceOptimization

class UserInterface:
    def __init__(self):
        self.dataset = None
        self.search_space = None
        self.evaluation_metrics = None
        self.progress = None
        self.results = None

    def input_dataset(self):
        dataset_handler = DatasetHandling()
        self.dataset = dataset_handler.import_dataset(file_format)
        self.dataset = dataset_handler.preprocess_dataset(augmentation_techniques, preprocessing_steps)

    def define_search_space(self):
        architecture_search = ArchitectureSearch()
        self.search_space = architecture_search.generate_architecture()

    def choose_evaluation_metrics(self):
        evaluation_metrics = EvaluationMetrics()
        self.evaluation_metrics = evaluation_metrics.choose_metrics()

    def display_progress(self):
        logging = ErrorHandlingLogging()
        self.progress = logging.log_progress()

    def display_results(self):
        results_visualization = ResultsVisualization()
        self.results = results_visualization.display_results()

# Instantiate the UserInterface class
user_interface = UserInterface()

# Test the code by calling the methods
user_interface.input_dataset()
user_interface.define_search_space()
user_interface.choose_evaluation_metrics()
user_interface.display_progress()
user_interface.display_results()