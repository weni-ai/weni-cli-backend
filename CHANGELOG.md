# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-03-27

### Added
- Add VersionCheckMiddleware for CLI version validation
- Enhance package installation and Sentry integration

## [1.1.0] - 2025-03-14

### Added

- Added permissions router and verify permission endpoint 
- Integrate Sentry SDK for error tracking and performance monitoring
- Update README.md with project features, setup instructions, and Docker usage

## [1.0.0] - 2025-03-08

### Added

- Initial release of the Weni CLI backend
- FastAPI-based REST API with version routing
- Health endpoints for service monitoring
- AWS Lambda client integration for:
  - Creating Lambda functions
  - Invoking Lambda functions
  - Managing function lifecycle (waiting for active status)
  - Deleting Lambda functions
- AWS CloudWatch Logs client for retrieving function execution logs
- Authentication middleware with project-based authorization
- Weni API client for authorization checks
- Skills run endpoints for testing skills
- Agents deployment endpoints
- Comprehensive testing suite with pytest
- Docker and Docker Compose support for local development
- CI/CD pipeline configuration
