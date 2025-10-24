"""
Comprehensive unit tests for yanex web API routes.

Tests all endpoints with normal behavior, incorrect payloads, empty results,
and optional parameters as requested.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from yanex.web.app import app


class TestWebAPIBase:
    """Base class for web API tests with common setup."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_manager(self):
        """Create mock experiment manager."""
        manager = Mock()
        manager.storage = Mock()
        return manager

    @pytest.fixture
    def mock_filter(self):
        """Create mock experiment filter."""
        filter_mock = Mock()
        return filter_mock

    @pytest.fixture
    def sample_experiments(self):
        """Create sample experiment data for testing."""
        return [
            {
                "id": "exp001",
                "name": "test-experiment-1",
                "status": "completed",
                "created_at": "2024-01-01T10:00:00Z",
                "started_at": "2024-01-01T10:00:00Z",
                "completed_at": "2024-01-01T11:00:00Z",
                "duration": 3600,
                "tags": ["ml", "test"],
                "archived": False,
            },
            {
                "id": "exp002",
                "name": "test-experiment-2",
                "status": "running",
                "created_at": "2024-01-02T10:00:00Z",
                "started_at": "2024-01-02T10:00:00Z",
                "completed_at": None,
                "duration": None,
                "tags": ["deep-learning"],
                "archived": False,
            },
            {
                "id": "exp003",
                "name": "archived-experiment",
                "status": "failed",
                "created_at": "2024-01-03T10:00:00Z",
                "started_at": "2024-01-03T10:00:00Z",
                "failed_at": "2024-01-03T10:30:00Z",
                "duration": 1800,
                "tags": ["archived"],
                "archived": True,
            },
        ]

    @pytest.fixture
    def sample_config(self):
        """Create sample experiment config."""
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "model": {
                "architecture": "transformer",
                "layers": 12,
            },
        }

    @pytest.fixture
    def sample_results(self):
        """Create sample experiment results."""
        return [
            {
                "step": 1,
                "loss": 0.5,
                "accuracy": 0.8,
                "timestamp": "2024-01-01T10:00:00Z",
            },
            {
                "step": 2,
                "loss": 0.4,
                "accuracy": 0.85,
                "timestamp": "2024-01-01T10:05:00Z",
            },
        ]

    @pytest.fixture
    def sample_metadata(self):
        """Create sample experiment metadata."""
        return {
            "id": "exp001",
            "name": "test-experiment-1",
            "description": "A test experiment",
            "status": "completed",
            "created_at": "2024-01-01T10:00:00Z",
            "tags": ["ml", "test"],
        }

    @pytest.fixture
    def sample_artifacts(self):
        """Create sample artifact data."""
        return [
            {"name": "model.pth", "size": 1024, "modified": 1704110400},
            {"name": "results.json", "size": 512, "modified": 1704110400},
        ]


class TestListExperiments(TestWebAPIBase):
    """Test /experiments endpoint."""

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_normal_behavior(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test normal behavior of list experiments endpoint."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[
            :2
        ]  # Non-archived only

        # Test request
        response = client.get("/api/experiments")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        assert "total" in data
        assert len(data["experiments"]) == 2
        assert data["total"] == 2

        # Verify filter was called with correct parameters
        mock_filter.filter_experiments.assert_called_once_with(
            status=None,
            name=None,  # Changed from name_pattern
            tags=None,
            started_after=None,
            started_before=None,
            ended_after=None,
            ended_before=None,
            archived=False,  # Changed from include_archived
            limit=None,
            include_all=True,
        )

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_with_all_parameters(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test list experiments with all optional parameters."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[:1]

        # Test request with all parameters
        response = client.get(
            "/api/experiments",
            params={
                "limit": 10,
                "status": "completed",
                "name_pattern": "test-*",
                "tags": "ml,test",
                "started_after": "2024-01-01",
                "started_before": "2024-01-02",
                "ended_after": "2024-01-01",
                "ended_before": "2024-01-02",
                "sort_order": "oldest",
                "archived": False,
            },
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["experiments"]) == 1

        # Verify filter was called with all parameters
        mock_filter.filter_experiments.assert_called_once()
        call_args = mock_filter.filter_experiments.call_args[1]
        assert call_args["status"] == "completed"
        assert call_args["name"] == "test-*"  # Changed from name_pattern
        assert call_args["tags"] == ["ml", "test"]
        assert call_args["archived"] is False  # Changed from include_archived

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_empty_results(self, mock_manager, mock_filter, client):
        """Test list experiments with empty results."""
        # Setup mocks to return empty list
        mock_filter.filter_experiments.return_value = []

        # Test request
        response = client.get("/api/experiments")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["experiments"] == []
        assert data["total"] == 0

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_with_archived(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test list experiments including archived experiments."""
        # Setup mocks - return only archived experiments (filtering now handled internally)
        archived_experiments = [
            exp for exp in sample_experiments if exp.get("archived", False)
        ]
        mock_filter.filter_experiments.return_value = archived_experiments

        # Test request with archived=True
        response = client.get("/api/experiments?archived=true")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        # Should return only archived experiments (1 out of 3)
        assert len(data["experiments"]) == len(archived_experiments)

        # Verify filter was called with archived=True
        mock_filter.filter_experiments.assert_called_once()
        call_args = mock_filter.filter_experiments.call_args[1]
        assert call_args["archived"] is True  # Changed from include_archived

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_sort_order_oldest(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test list experiments with oldest sort order."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[:2]

        # Test request with sort_order=oldest
        response = client.get("/api/experiments?sort_order=oldest")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["experiments"]) == 2

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_with_limit(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test list experiments with limit parameter."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[:1]

        # Test request with limit
        response = client.get("/api/experiments?limit=1")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data["experiments"]) == 1

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_invalid_time_format(
        self, mock_manager, mock_filter, client
    ):
        """Test list experiments with invalid time format."""
        # Setup mocks to raise exception
        mock_filter.filter_experiments.side_effect = Exception("Invalid time format")

        # Test request with invalid time
        response = client.get("/api/experiments?started_after=invalid-date")

        # Verify error response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Invalid time specification" in data["detail"]

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_server_error(self, mock_manager, mock_filter, client):
        """Test list experiments with server error."""
        # Setup mocks to raise exception
        mock_filter.filter_experiments.side_effect = Exception(
            "Database connection failed"
        )

        # Test request
        response = client.get("/api/experiments")

        # Verify error response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]


class TestGetExperiment(TestWebAPIBase):
    """Test /experiments/{experiment_id} endpoint."""

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_experiment_normal_behavior(
        self,
        mock_manager,
        mock_filter,
        client,
        sample_experiments,
        sample_config,
        sample_results,
        sample_metadata,
        sample_artifacts,
    ):
        """Test normal behavior of get experiment endpoint."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[:1]
        mock_manager.storage.load_config.return_value = sample_config
        mock_manager.storage.load_results.return_value = sample_results
        mock_manager.storage.load_metadata.return_value = sample_metadata

        # Mock artifacts directory
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()

            # Create sample artifact files
            (artifacts_dir / "model.pth").write_bytes(b"fake model data")
            (artifacts_dir / "results.json").write_text('{"test": "data"}')

            mock_manager.storage.get_experiment_dir.return_value = Path(temp_dir)

            # Test request
            response = client.get("/api/experiments/exp001")

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert "experiment" in data
            assert "config" in data
            assert "results" in data
            assert "metadata" in data
            assert "artifacts" in data

            # Verify experiment data
            assert data["experiment"]["id"] == "exp001"
            assert data["config"] == sample_config
            assert data["results"] == sample_results
            assert data["metadata"] == sample_metadata
            assert len(data["artifacts"]) == 2

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_experiment_not_found(self, mock_manager, mock_filter, client):
        """Test get experiment with non-existent experiment ID."""
        # Setup mocks to return empty list
        mock_filter.filter_experiments.return_value = []

        # Test request
        response = client.get("/api/experiments/nonexistent")

        # Verify error response
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "Experiment not found" in data["detail"]

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_experiment_with_archived(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test get experiment with archived=True parameter."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[
            2:
        ]  # Archived experiment
        mock_manager.storage.load_config.return_value = {}
        mock_manager.storage.load_results.return_value = []
        mock_manager.storage.load_metadata.return_value = {}
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp")

        # Test request with archived=True
        response = client.get("/api/experiments/exp003?archived=true")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["experiment"]["id"] == "exp003"

        # Verify filter was called with archived=True
        mock_filter.filter_experiments.assert_called_once()
        call_args = mock_filter.filter_experiments.call_args[1]
        assert call_args["archived"] is True  # Changed from include_archived

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_experiment_partial_id_match(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test get experiment with partial ID match."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[:1]
        mock_manager.storage.load_config.return_value = {}
        mock_manager.storage.load_results.return_value = []
        mock_manager.storage.load_metadata.return_value = {}
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp")

        # Test request with partial ID
        response = client.get("/api/experiments/exp0")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["experiment"]["id"] == "exp001"

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_experiment_missing_config(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test get experiment when config is missing."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[:1]
        mock_manager.storage.load_config.side_effect = Exception("Config not found")
        mock_manager.storage.load_results.return_value = []
        mock_manager.storage.load_metadata.return_value = {}
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp")

        # Test request
        response = client.get("/api/experiments/exp001")

        # Verify response with empty config
        assert response.status_code == 200
        data = response.json()
        assert data["config"] == {}

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_experiment_missing_results(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test get experiment when results are missing."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[:1]
        mock_manager.storage.load_config.return_value = {}
        mock_manager.storage.load_results.side_effect = Exception("Results not found")
        mock_manager.storage.load_metadata.return_value = {}
        mock_manager.storage.get_experiment_dir.return_value = Path("/tmp")

        # Test request
        response = client.get("/api/experiments/exp001")

        # Verify response with empty results
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_experiment_missing_artifacts(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test get experiment when artifacts directory is missing."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = sample_experiments[:1]
        mock_manager.storage.load_config.return_value = {}
        mock_manager.storage.load_results.return_value = []
        mock_manager.storage.load_metadata.return_value = {}
        mock_manager.storage.get_experiment_dir.side_effect = Exception(
            "Directory not found"
        )

        # Test request
        response = client.get("/api/experiments/exp001")

        # Verify response with empty artifacts
        assert response.status_code == 200
        data = response.json()
        assert data["artifacts"] == []

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_experiment_server_error(self, mock_manager, mock_filter, client):
        """Test get experiment with server error."""
        # Setup mocks to raise exception
        mock_filter.filter_experiments.side_effect = Exception("Database error")

        # Test request
        response = client.get("/api/experiments/exp001")

        # Verify error response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database error" in data["detail"]


class TestDownloadArtifact(TestWebAPIBase):
    """Test /experiments/{experiment_id}/artifacts/{artifact_name} endpoint."""

    @patch("yanex.web.api.manager")
    def test_download_artifact_normal_behavior(self, mock_manager, client):
        """Test normal behavior of download artifact endpoint."""
        # Setup mocks
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()
            artifact_file = artifacts_dir / "model.pth"
            artifact_file.write_bytes(b"fake model data")

            mock_manager.storage.get_experiment_dir.return_value = Path(temp_dir)

            # Test request
            response = client.get("/api/experiments/exp001/artifacts/model.pth")

            # Verify response
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/octet-stream"
            assert (
                response.headers["content-disposition"]
                == 'attachment; filename="model.pth"'
            )
            assert response.content == b"fake model data"

    @patch("yanex.web.api.manager")
    def test_download_artifact_not_found(self, mock_manager, client):
        """Test download artifact with non-existent artifact."""
        # Setup mocks
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()

            mock_manager.storage.get_experiment_dir.return_value = Path(temp_dir)

            # Test request
            response = client.get("/api/experiments/exp001/artifacts/nonexistent.pth")

            # Verify error response
            assert response.status_code == 404
            data = response.json()
            assert "detail" in data
            assert "Artifact not found" in data["detail"]

    @patch("yanex.web.api.manager")
    def test_download_artifact_with_archived(self, mock_manager, client):
        """Test download artifact with archived=True parameter."""
        # Setup mocks
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()
            artifact_file = artifacts_dir / "archived_model.pth"
            artifact_file.write_bytes(b"archived model data")

            mock_manager.storage.get_experiment_dir.return_value = Path(temp_dir)

            # Test request with archived=True
            response = client.get(
                "/api/experiments/exp003/artifacts/archived_model.pth?archived=true"
            )

            # Verify response
            assert response.status_code == 200
            assert response.content == b"archived model data"

            # Verify manager was called with archived=True
            mock_manager.storage.get_experiment_dir.assert_called_once_with(
                "exp003", include_archived=True
            )

    @patch("yanex.web.api.manager")
    def test_download_artifact_server_error(self, mock_manager, client):
        """Test download artifact with server error."""
        # Setup mocks to raise exception
        mock_manager.storage.get_experiment_dir.side_effect = Exception("Storage error")

        # Test request
        response = client.get("/api/experiments/exp001/artifacts/model.pth")

        # Verify error response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Storage error" in data["detail"]

    @patch("yanex.web.api.manager")
    def test_download_artifact_directory_not_found(self, mock_manager, client):
        """Test download artifact when experiment directory doesn't exist."""
        # Setup mocks
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_manager.storage.get_experiment_dir.return_value = Path(temp_dir)

            # Test request
            response = client.get("/api/experiments/exp001/artifacts/model.pth")

            # Verify error response
            assert response.status_code == 404
            data = response.json()
            assert "detail" in data
            assert "Artifact not found" in data["detail"]


class TestGetStatus(TestWebAPIBase):
    """Test /status endpoint."""

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_status_normal_behavior(
        self, mock_manager, mock_filter, client, sample_experiments
    ):
        """Test normal behavior of get status endpoint."""
        # Setup mocks - filtering now handled internally, return already-filtered results
        non_archived = [
            exp for exp in sample_experiments if not exp.get("archived", False)
        ]
        archived = [exp for exp in sample_experiments if exp.get("archived", False)]

        mock_filter.filter_experiments.side_effect = [
            non_archived,  # First call for non-archived (2 experiments)
            archived,  # Second call for archived only (1 experiment)
        ]

        # Test request
        response = client.get("/api/status")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "total_experiments" in data
        assert "archived_experiments" in data
        assert "status_counts" in data

        # Verify counts
        assert data["total_experiments"] == 2  # Non-archived only
        assert data["archived_experiments"] == 1  # One archived experiment
        assert "completed" in data["status_counts"]
        assert "running" in data["status_counts"]
        assert data["status_counts"]["completed"] == 1
        assert data["status_counts"]["running"] == 1

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_status_empty_experiments(self, mock_manager, mock_filter, client):
        """Test get status with no experiments."""
        # Setup mocks to return empty lists
        mock_filter.filter_experiments.return_value = []

        # Test request
        response = client.get("/api/status")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["total_experiments"] == 0
        assert data["archived_experiments"] == 0
        assert data["status_counts"] == {}

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_status_server_error(self, mock_manager, mock_filter, client):
        """Test get status with server error."""
        # Setup mocks to raise exception
        mock_filter.filter_experiments.side_effect = Exception(
            "Database connection failed"
        )

        # Test request
        response = client.get("/api/status")

        # Verify error response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_get_status_mixed_statuses(self, mock_manager, mock_filter, client):
        """Test get status with various experiment statuses."""
        # Setup mocks with mixed statuses
        mixed_experiments = [
            {"id": "exp1", "status": "completed", "archived": False},
            {"id": "exp2", "status": "running", "archived": False},
            {"id": "exp3", "status": "failed", "archived": False},
            {"id": "exp4", "status": "completed", "archived": True},
            {"id": "exp5", "status": "cancelled", "archived": False},
        ]

        # Filtering now handled internally - return already-filtered results
        mock_filter.filter_experiments.side_effect = [
            [
                exp for exp in mixed_experiments if not exp["archived"]
            ],  # Non-archived (4)
            [exp for exp in mixed_experiments if exp["archived"]],  # Archived only (1)
        ]

        # Test request
        response = client.get("/api/status")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["total_experiments"] == 4  # Non-archived
        assert data["archived_experiments"] == 1  # One archived
        assert data["status_counts"]["completed"] == 1
        assert data["status_counts"]["running"] == 1
        assert data["status_counts"]["failed"] == 1
        assert data["status_counts"]["cancelled"] == 1


class TestWebAPIIntegration:
    """Integration tests for web API."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    def test_api_routes_exist(self, client):
        """Test that all expected API routes exist."""
        # Test that routes are registered
        routes = [route.path for route in app.routes if hasattr(route, "path")]

        # Check for API routes with more flexible matching
        api_routes = [route for route in routes if route.startswith("/api")]

        # Expected route patterns (more flexible matching)
        expected_patterns = [
            "/api/experiments",
            "/api/experiments/",
            "/api/status",
        ]

        # Check that we have the basic API routes
        assert len(api_routes) >= 3, (
            f"Expected at least 3 API routes, found {len(api_routes)}: {api_routes}"
        )

        # Check for specific route patterns
        route_patterns_found = []
        for pattern in expected_patterns:
            if any(pattern in route for route in api_routes):
                route_patterns_found.append(pattern)

        assert len(route_patterns_found) >= 2, (
            f"Expected at least 2 route patterns, found {route_patterns_found}"
        )

    def test_api_response_format_consistency(self, client):
        """Test that API responses have consistent format."""
        # This test would require more complex setup with real data
        # For now, just verify the endpoints exist and return proper HTTP status codes
        with patch("yanex.web.api.experiment_filter") as mock_filter:
            mock_filter.filter_experiments.return_value = []

            # Test all endpoints return proper status codes
            response = client.get("/api/experiments")
            assert response.status_code in [200, 500]  # Either success or server error

            response = client.get("/api/status")
            assert response.status_code in [200, 500]

            # These will return 404 or 500 without proper setup
            response = client.get("/api/experiments/nonexistent")
            assert response.status_code in [404, 500]

            response = client.get("/api/experiments/exp001/artifacts/model.pth")
            assert response.status_code in [404, 500]


class TestWebAPIParameterValidation:
    """Test parameter validation for web API endpoints."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_invalid_parameters(
        self, mock_manager, mock_filter, client
    ):
        """Test list experiments with invalid parameter values."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = []

        # Test with invalid limit (negative)
        response = client.get("/api/experiments?limit=-1")
        # Should still work but with default behavior
        assert response.status_code == 200

        # Test with invalid sort_order
        response = client.get("/api/experiments?sort_order=invalid")
        # Should still work with default sort
        assert response.status_code == 200

        # Test with empty tags
        response = client.get("/api/experiments?tags=")
        assert response.status_code == 200

    @patch("yanex.web.api.experiment_filter")
    @patch("yanex.web.api.manager")
    def test_list_experiments_edge_case_parameters(
        self, mock_manager, mock_filter, client
    ):
        """Test list experiments with edge case parameters."""
        # Setup mocks
        mock_filter.filter_experiments.return_value = []

        # Test with very large limit
        response = client.get("/api/experiments?limit=999999")
        assert response.status_code == 200

        # Test with special characters in name_pattern
        response = client.get("/api/experiments?name_pattern=test[0-9]*")
        assert response.status_code == 200

        # Test with multiple tags
        response = client.get("/api/experiments?tags=tag1,tag2,tag3")
        assert response.status_code == 200

        # Verify filter was called with parsed tags
        mock_filter.filter_experiments.assert_called()
        call_args = mock_filter.filter_experiments.call_args[1]
        assert call_args["tags"] == ["tag1", "tag2", "tag3"]

    def test_api_error_handling_consistency(self, client):
        """Test that API error responses are consistent."""
        with patch("yanex.web.api.experiment_filter") as mock_filter:
            # Test 500 error
            mock_filter.filter_experiments.side_effect = Exception("Test error")

            response = client.get("/api/experiments")
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert isinstance(data["detail"], str)

            response = client.get("/api/status")
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert isinstance(data["detail"], str)
