import zipfile
from collections.abc import Generator
from io import BytesIO
from typing import Any

import pytest

from app.services.agents.active.models import ActiveAgentResourceModel
from app.services.agents.active.processor import ActiveAgentProcessor


@pytest.fixture
def active_agent_resource_model_fixture() -> ActiveAgentResourceModel:
    # Create a dummy preprocessor zip
    preprocessor_content = b"preprocessor_dummy_content"
    preprocessor_zip = BytesIO()
    with zipfile.ZipFile(preprocessor_zip, "w") as zf:
        zf.writestr("preprocessor.py", preprocessor_content)
        zf.writestr("requirements.txt", b"dependency1==1.0")
    preprocessor_zip.seek(0)

    # Create a dummy rule zip
    rule_content = b"rule_dummy_content"
    rule_zip = BytesIO()
    with zipfile.ZipFile(rule_zip, "w") as zf:
        zf.writestr("rule.py", rule_content)
        zf.writestr("requirements.txt", b"dependency2==2.0")
    rule_zip.seek(0)

    return ActiveAgentResourceModel(
        preprocessor=preprocessor_zip.getvalue(), rules={"rule1": rule_zip.getvalue()}, preprocessor_example=None
    )


def test_active_agent_processor_process_creates_zip_file(
    active_agent_resource_model_fixture: ActiveAgentResourceModel, mocker: Any
) -> None:
    project_uuid = "test_project_uuid"
    toolkit_version = "0.1.0"
    agent_key = "test_agent"

    # Mock the Packager class where it's imported in the processor module
    packager_mock_class = mocker.patch("app.services.agents.active.processor.Packager")
    # Get the mock instance that will be created inside the process method
    packager_mock_instance = packager_mock_class.return_value
    packager_mock_instance.build_lambda_function_file.return_value = "lambda_function_content"

    processor = ActiveAgentProcessor(
        project_uuid=project_uuid, toolkit_version=toolkit_version, agent_resource=active_agent_resource_model_fixture
    )

    output_zip = processor.process(agent_key)

    assert isinstance(output_zip, BytesIO)
    assert output_zip.name == f"{agent_key}.zip"

    # Verify zip content (basic check)
    with zipfile.ZipFile(output_zip) as zf:
        assert "lambda_function.py" in zf.namelist()
        assert "preprocessor/preprocessor.py" in zf.namelist()
        assert "preprocessor/requirements.txt" in zf.namelist()
        assert "rules/rule1/rule.py" in zf.namelist()
        assert "rules/rule1/requirements.txt" in zf.namelist()

    # Check constructor was called (implicitly checks if Packager was instantiated)
    packager_mock_class.assert_called_once()
    # Check method calls on the instance
    packager_mock_instance.install_packages.assert_called_once()
    # Further checks on arguments can be added here (e.g., check package names)
    assert packager_mock_instance.install_dependencies.call_count == 2
    # Further checks on arguments can be added here (e.g., check requirements paths)
    packager_mock_instance.build_lambda_function_file.assert_called_once()
    # Further checks on arguments can be added here


@pytest.fixture(autouse=True)
def mock_shutil_rmtree(mocker: Any) -> Generator[Any, None, None]:
    yield mocker.patch("shutil.rmtree")


def test_active_agent_processor_process_cleanup(
    active_agent_resource_model_fixture: ActiveAgentResourceModel, mocker: Any, mock_shutil_rmtree: Any
) -> None:
    project_uuid = "test_project_uuid_cleanup"
    toolkit_version = "0.1.0"
    agent_key = "test_agent_cleanup"

    # Mock the Packager class
    packager_mock_class = mocker.patch("app.services.agents.active.processor.Packager")
    packager_mock_instance = packager_mock_class.return_value
    packager_mock_instance.build_lambda_function_file.return_value = "lambda_function_content"

    # Mock mkdtemp to control and check the path used
    mock_mkdtemp = mocker.patch("tempfile.mkdtemp")
    # Use a fixed string path instead of TemporaryDirectory
    test_temp_dir = f"/tmp/fake_agent_{agent_key}_{project_uuid}_"  # Fake path
    mock_mkdtemp.return_value = test_temp_dir
    # Mock Path.exists to ensure cleanup is attempted
    mocker.patch("app.services.agents.active.processor.Path.exists", return_value=True)
    # Mock Path.mkdir to prevent failure when using fake temp path
    mocker.patch("app.services.agents.active.processor.Path.mkdir")

    processor = ActiveAgentProcessor(
        project_uuid=project_uuid, toolkit_version=toolkit_version, agent_resource=active_agent_resource_model_fixture
    )

    try:
        processor.process(agent_key)
    finally:
        mock_mkdtemp.assert_called_once_with(prefix=f"agent_{agent_key}_{project_uuid}_")
        mock_shutil_rmtree.assert_called_with(test_temp_dir, ignore_errors=True)
        # No cleanup needed for the fake path


def test_active_agent_processor_process_handles_exception(
    active_agent_resource_model_fixture: ActiveAgentResourceModel, mocker: Any, mock_shutil_rmtree: Any
) -> None:
    project_uuid = "test_project_uuid_exception"
    toolkit_version = "0.1.0"
    agent_key = "test_agent_exception"

    # Mock the Packager class
    packager_mock_class = mocker.patch("app.services.agents.active.processor.Packager")
    packager_mock_instance = packager_mock_class.return_value
    # Make install_packages raise an exception
    packager_mock_instance.install_packages.side_effect = Exception("Install error")
    packager_mock_instance.build_lambda_function_file.return_value = "lambda_function_content"

    mock_mkdtemp = mocker.patch("tempfile.mkdtemp")
    # Use a fixed string path instead of TemporaryDirectory
    test_temp_dir = f"/tmp/fake_agent_{agent_key}_{project_uuid}_"  # Fake path
    mock_mkdtemp.return_value = test_temp_dir
    # Mock Path.mkdir to prevent failure before install_packages is called
    mocker.patch("app.services.agents.active.processor.Path.mkdir")
    # Mock Path.exists to ensure cleanup is attempted
    mocker.patch("app.services.agents.active.processor.Path.exists", return_value=True)

    processor = ActiveAgentProcessor(
        project_uuid=project_uuid, toolkit_version=toolkit_version, agent_resource=active_agent_resource_model_fixture
    )

    with pytest.raises(Exception, match="Install error"):
        processor.process(agent_key)

    mock_mkdtemp.assert_called_once_with(prefix=f"agent_{agent_key}_{project_uuid}_")
    mock_shutil_rmtree.assert_called_with(test_temp_dir, ignore_errors=True)
    # No cleanup needed for the fake path
