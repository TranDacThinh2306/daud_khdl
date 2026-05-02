# """
# test_models.py - Tests for model training and evaluation
# ==========================================================
# """

# import pytest
# import numpy as np
# from sklearn.datasets import make_classification

# from src.models.train import ModelTrainer
# from src.models.evaluate import ModelEvaluator
# from src.models.architectures.random_forest import build_random_forest
# from src.models.architectures.svm import build_svm
# from src.models.architectures.xgboost import build_xgboost


# @pytest.fixture
# def sample_data():
#     """Generate sample classification data."""
#     X, y = make_classification(
#         n_samples=200, n_features=20, n_informative=10,
#         n_classes=2, random_state=42
#     )
#     return X, y


# class TestModelTrainer:
#     """Tests for ModelTrainer."""

#     def test_train_random_forest(self, sample_data):
#         X, y = sample_data
#         model = build_random_forest(n_estimators=10)
#         trainer = ModelTrainer()
#         trained = trainer.train(model, X, y)
#         predictions = trained.predict(X)
#         assert len(predictions) == len(y)

#     def test_cross_validate(self, sample_data):
#         X, y = sample_data
#         model = build_random_forest(n_estimators=10)
#         trainer = ModelTrainer(n_folds=3)
#         results = trainer.cross_validate(model, X, y, scoring=["accuracy", "f1"])
#         assert "test_accuracy" in results
#         assert "test_f1" in results


# class TestModelEvaluator:
#     """Tests for ModelEvaluator."""

#     def test_evaluate(self, sample_data):
#         X, y = sample_data
#         model = build_random_forest(n_estimators=10)
#         model.fit(X, y)

#         evaluator = ModelEvaluator()
#         metrics = evaluator.evaluate(model, X, y)
#         assert "accuracy" in metrics
#         assert "f1" in metrics
#         assert 0 <= metrics["accuracy"] <= 1

#     def test_compare_models(self, sample_data):
#         X, y = sample_data
#         rf = build_random_forest(n_estimators=10)
#         rf.fit(X, y)

#         evaluator = ModelEvaluator()
#         evaluator.evaluate(rf, X, y, model_name="RF")
#         comparison = evaluator.compare_models()
#         assert "RF" in comparison


# class TestArchitectures:
#     """Tests for model architectures."""

#     def test_build_random_forest(self):
#         model = build_random_forest()
#         assert model.n_estimators == 200

#     def test_build_svm(self):
#         model = build_svm()
#         assert hasattr(model, "fit")
#         assert hasattr(model, "predict")

#     def test_build_xgboost(self):
#         model = build_xgboost()
#         assert hasattr(model, "fit")
