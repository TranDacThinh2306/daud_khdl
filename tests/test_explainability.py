# """
# test_explainability.py - Kiểm tra stability của SHAP/LIME
# ============================================================
# Tests for XAI explanation generation and stability.
# """

# import pytest
# import numpy as np
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier

# from src.explainability.shap_explainer import SHAPExplainer
# from src.explainability.lime_explainer import LIMEExplainer
# from src.explainability.stability import StabilityChecker


# @pytest.fixture
# def trained_model_and_data():
#     """Create a trained model with sample data."""
#     X, y = make_classification(
#         n_samples=200, n_features=10, n_classes=2, random_state=42
#     )
#     model = RandomForestClassifier(n_estimators=50, random_state=42)
#     model.fit(X, y)
#     feature_names = [f"feature_{i}" for i in range(10)]
#     return model, X, y, feature_names


# class TestSHAPExplainer:
#     """Tests for SHAPExplainer."""

#     def test_fit_and_explain(self, trained_model_and_data):
#         model, X, y, feature_names = trained_model_and_data
#         explainer = SHAPExplainer(model, explainer_type="tree")
#         explainer.fit(X[:50])
#         shap_values = explainer.explain(X[:5])
#         assert shap_values is not None

#     def test_explain_single(self, trained_model_and_data):
#         model, X, y, feature_names = trained_model_and_data
#         explainer = SHAPExplainer(model, explainer_type="tree")
#         explainer.fit(X[:50])
#         result = explainer.explain_single(X[0], feature_names)
#         assert isinstance(result, dict)
#         assert len(result) == len(feature_names)

#     def test_feature_importance(self, trained_model_and_data):
#         model, X, y, feature_names = trained_model_and_data
#         explainer = SHAPExplainer(model, explainer_type="tree")
#         explainer.fit(X[:50])
#         importance = explainer.get_feature_importance(X[:20], feature_names)
#         assert isinstance(importance, dict)


# class TestLIMEExplainer:
#     """Tests for LIMEExplainer."""

#     def test_explain_instance(self, trained_model_and_data):
#         model, X, y, feature_names = trained_model_and_data
#         explainer = LIMEExplainer(X, feature_names=feature_names)
#         explanation = explainer.explain_instance(
#             X[0], model.predict_proba, num_features=5
#         )
#         assert explanation is not None

#     def test_get_top_features(self, trained_model_and_data):
#         model, X, y, feature_names = trained_model_and_data
#         explainer = LIMEExplainer(X, feature_names=feature_names)
#         explanation = explainer.explain_instance(X[0], model.predict_proba)
#         top = explainer.get_top_features(explanation)
#         assert isinstance(top, dict)


# class TestStabilityChecker:
#     """Tests for explanation stability."""

#     def test_lime_stability(self, trained_model_and_data):
#         model, X, y, feature_names = trained_model_and_data
#         lime_exp = LIMEExplainer(X, feature_names=feature_names)
#         checker = StabilityChecker(n_repeats=3)
#         result = checker.check_lime_stability(
#             X[0], lime_exp, model.predict_proba
#         )
#         assert "jaccard_similarity" in result
#         assert "is_stable" in result
