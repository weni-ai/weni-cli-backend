# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.1] - 2025-05-28

- feat: Add project to context in lambda_function.py.template

## [1.7.0] - 2025-05-27

- fix: Remove unused language field from lambda_function.py.template
- feat: pass contact to skill context

## [1.6.0] - 2025-05-21

### Added
- Feat: Gallery client 
- feat: Add ConfigureAgentsRequestModel and update agent configuration endpoint 
- feat: Introduce PassiveAgentConfigurator for agent configuration and processing 
- feat: Implement ActiveAgentConfigurator and related components for agent processing 
- Refactor package installation services 
- feat: Implement GalleryClient for agent management and add corresponding tests 

## [1.5.1] - 2025-05-08

### Added
- Enhance error handling in get_logs endpoint

## [1.5.0] - 2025-05-07

### Added
- Feat: add AWS region name to config

## [1.4.0] - 2025-05-06

### Added
- Feat: logs router 

## [1.3.0] - 2025-04-24

### Refactored
- Reorganize sentry import
- Rename skill-related components to tool-related
- Update agent and tool naming conventions in requests and routers
- Update logging and error messages to reflect tool terminology 

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
