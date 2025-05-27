import logging
import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from app.services.agents.active.models import ActiveAgentResourceModel
from app.services.package.package import Package, Packager

logger = logging.getLogger(__name__)


class ActiveAgentProcessor:
    def __init__(self, project_uuid: str, toolkit_version: str, agent_resource: ActiveAgentResourceModel):
        self.project_uuid = project_uuid
        self.toolkit_version = toolkit_version
        self.preprocessor = agent_resource.preprocessor
        self.rules = agent_resource.rules

    def process(self, agent_key: str) -> BytesIO:  # noqa: PLR0915
        temp_dir = None
        temp_prefix = f"agent_{agent_key}_{self.project_uuid}_"

        if not self.preprocessor:
            raise Exception("Preprocessor is required")

        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp(prefix=temp_prefix)
            temp_path = Path(temp_dir)

            # 1. create the package directory for any dependencies installation
            package_dir = temp_path / "package"
            package_dir.mkdir(exist_ok=True)

            # Install toolkit and sentry in package directory
            default_packages = [
                Package("weni-agents-toolkit", self.toolkit_version),
                Package("sentry-sdk", "2.24.1"),
            ]

            packager = Packager(package_dir)
            packager.install_packages(default_packages)

            # 2. create the preprocessor folder, installing dependencies if needed
            preprocessor_path = temp_path / "preprocessor"
            preprocessor_path.mkdir(exist_ok=True)

            # Extract preprocessor zip content
            preprocessor_zip = BytesIO(self.preprocessor.content)
            with zipfile.ZipFile(preprocessor_zip) as zip_ref:
                logger.debug(f"Extracting {len(zip_ref.namelist())} files to {preprocessor_path}")
                zip_ref.extractall(preprocessor_path)

            # install preprocessor dependencies
            requirements_path = preprocessor_path / "requirements.txt"
            if requirements_path.exists():
                packager.install_dependencies(requirements_path, "preprocessor")

            # 3. create each rule folder
            for rule in self.rules:
                rule_path = temp_path / "rules" / rule.key
                rule_path.mkdir(exist_ok=True, parents=True)

                # Extract rule zip content
                rule_zip = BytesIO(rule.content)
                with zipfile.ZipFile(rule_zip) as zip_ref:
                    logger.debug(f"Extracting {len(zip_ref.namelist())} files to {rule_path}")
                    zip_ref.extractall(rule_path)

                # install rule dependencies
                requirements_path = rule_path / "requirements.txt"
                if requirements_path.exists():
                    packager.install_dependencies(requirements_path, f"rule_{rule.class_name}")

            # 4. create the lambda function formatting with preprocessor and rules imports/initializations
            logger.info(f"Creating Lambda function for {agent_key}")

            template_path = Path(__file__).parent / "templates" / "lambda_function.py.template"
            replacements = {
                "preprocessor_module": self.preprocessor.module_name,
                "preprocessor_class": self.preprocessor.class_name,
                "official_rules_imports": self.mount_rule_imports(),
                "rule_class_to_template": str(self.mount_rule_class_to_template_map()),
                "official_rules_instances": self.mount_rule_instances_list(),
                "classname_to_key_map": str(self.mount_rule_classname_to_key_map()),
            }
            lambda_function_content = packager.build_lambda_function_file(template_path, replacements)
            lambda_function_path = temp_path / "lambda_function.py"
            with open(lambda_function_path, "w") as f:
                f.write(lambda_function_content)

            # 5. create the output zip
            logger.info(f"Creating output zip for {agent_key}")
            output_buffer = BytesIO()
            with zipfile.ZipFile(output_buffer, "w", zipfile.ZIP_DEFLATED) as zip_out:
                # Walk through all files in the temp directory
                for root, _, files in os.walk(temp_dir):
                    root_path = Path(root)
                    for file in files:
                        file_path = root_path / file
                        arc_name = file_path.relative_to(temp_path)

                        if str(arc_name).startswith("package/"):
                            # move content out of /package and insert it at root
                            zip_path = str(arc_name).replace("package/", "", 1)
                        else:
                            # keep the same path
                            zip_path = str(arc_name)

                        zip_out.write(file_path, arcname=zip_path)

            # Prepare the output
            output_buffer.seek(0)
            output_buffer.name = f"{agent_key}.zip"
            logger.info(f"Completed creation of agent zip for {agent_key}")
            return output_buffer

        except Exception as e:
            logger.error(f"Error creating creating active agent {agent_key} for project {self.project_uuid}: {e}")
            raise e

        finally:
            # Clean up the temporary directory
            if temp_dir and Path(temp_dir).exists():
                logger.debug(f"Cleaning up temporary directory {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)

    def mount_rule_imports(self) -> str:
        return "\n".join([f"from rules.{rule.key}.{rule.module_name} import {rule.class_name}" for rule in self.rules])

    def mount_rule_class_to_template_map(self) -> dict[str, str]:
        return {f"{rule.class_name}": f"{rule.template}" for rule in self.rules}

    def mount_rule_instances_list(self) -> str:
        return ", ".join([f"{rule.class_name}()" for rule in self.rules])

    def mount_rule_classname_to_key_map(self) -> dict[str, str]:
        return {f"{rule.class_name}": f"{rule.key}" for rule in self.rules}
