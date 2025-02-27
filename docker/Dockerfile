# syntax = docker/dockerfile:1

ARG PYTHON_VERSION="3.12"
ARG DEBIAN_VERSION="bookworm"
ARG POETRY_VERSION="1.8.5"

ARG BUILD_DEPS="\
  gcc bzip2 git curl libpq-dev gettext \
  libgdal-dev python3-cffi python3-gdal \
  python3-dev default-libmysqlclient-dev build-essential \
  build-essential \
  git cmake \
  autoconf pkg-config autoconf libtool automake \
  libmariadb-dev npm"
# default-libmysqlclient-dev
ARG RUNTIME_DEPS="\
  git \
  tzdata \
  netcat-traditional \
  curl \
  gosu"
#libmariadb3 \

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS base

ARG POETRY_VERSION

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  DEBIAN_FRONTEND=noninteractive \
  APP=rapidpro \
  APP_PATH=/app \
  APP_USER=app_user \
  APP_GROUP=app_group \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PATH="/opt/npm-globals/bin:/app/node_modules/.bin/:${PATH}:/install/bin"

LABEL os="debian" \
  os.version="12" \
  name="cli-backend" \
  description="Backend for CLI image" \
  maintainer="https://github.com/weni-ai" \
  org.opencontainers.image.url="https://github.com/weni-ai/weni-cli-backend" \
  org.opencontainers.image.documentation="https://github.com/weni-ai/weni-cli-backend" \
  org.opencontainers.image.source="https://github.com/weni-ai/weni-cli-backend" \
  org.opencontainers.image.title="cli-backend"

RUN addgroup --gid 1999 "${APP_GROUP}" \
  && useradd --system -m -d "${APP_PATH}" -u 1999 -g 1999 "${APP_USER}"

WORKDIR "${APP_PATH}"

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

FROM base AS build-poetry

ARG POETRY_VERSION

COPY pyproject.toml poetry.lock .

RUN --mount=type=cache,mode=0755,target=/pip_cache,id=pip pip install --cache-dir /pip_cache -U poetry=="${POETRY_VERSION}" \
  && poetry cache clear -n --all pypi \
  && poetry export --without-hashes --output requirements.txt

FROM base AS build

ARG BUILD_DEPS

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get install --no-install-recommends --no-install-suggests -y ${BUILD_DEPS}

COPY --from=build-poetry /app/requirements.txt /tmp/dep/
RUN --mount=type=cache,mode=0755,target=/pip_cache,id=pip pip install --cache-dir /pip_cache --prefix=/install -r /tmp/dep/requirements.txt

FROM base

ARG BUILD_DEPS
ARG RUNTIME_DEPS

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && SUDO_FORCE_REMOVE=yes apt-get remove --purge -y ${BUILD_DEPS} \
  && apt-get autoremove -y \
  && apt-get install -y --no-install-recommends ${RUNTIME_DEPS} \
  && rm -rf /usr/share/man /usr/share/doc

COPY --from=build /install /usr/local
COPY --chown=${APP_USER}:${APP_GROUP} . ${APP_PATH}
COPY --chown=${APP_USER}:${APP_GROUP} docker/docker-entrypoint.sh /

USER "${APP_USER}:${APP_GROUP}"
EXPOSE 8000
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["start"]
