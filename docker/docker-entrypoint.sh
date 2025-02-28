#!/bin/bash

export WSGI_APP=${WSGI_APP:-"app.main:app"}
export CELERY_APP=${CELERY_APP:-"${WSGI_APP}"}
export LOG_LEVEL=${LOG_LEVEL:-"info"}
export FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-'*'}
export CELERY_MAX_WORKERS=${CELERY_MAX_WORKERS:-'4'}
export CELERY_BEAT_DATABASE_FILE=${CELERY_BEAT_DATABASE_FILE:-'/tmp/celery_beat_database'}
export WORKERS=${WORKERS:-'8'}
#set -o errexit
#set -o pipefail
#set -o nounset

do_gosu(){
    user="$1"
    shift 1

    is_exec="false"
    if [ "$1" = "exec" ]; then
        is_exec="true"
        shift 1
    fi

    if [ "$(id -u)" = "0" ]; then
        if [ "${is_exec}" = "true" ]; then
            exec gosu "${user}" "$@"
        else
            gosu "${user}" "$@"
            return "$?"
        fi
    else
        if [ "${is_exec}" = "true" ]; then
            exec "$@"
        else
            eval '"$@"'
            return "$?"
        fi
    fi
}


if [[ "start" == "$1" ]]; then
    echo "Starting server..."
    do_gosu "${APP_USER}:${APP_GROUP}" exec uvicorn "${WSGI_APP}" \
      --host 0.0.0.0 \
      --limit-max-requests 10000 \
      --forwarded-allow-ips "${FORWARDED_ALLOW_IPS}" \
      --log-level "${LOG_LEVEL}" \
      --use-colors \
      --no-server-header \
      --no-date-header \
      --workers "${WORKERS}"
elif [[ "celery-worker" == "$1" ]]; then
    celery_queue="celery"
    if [ "${2}" ] ; then
        celery_queue="${2}"
    fi
    do_gosu "${APP_USER}:${APP_GROUP}" exec celery \
        -A "${CELERY_APP}" --workdir="${APP_PATH}" worker \
        -Q "${celery_queue}" \
        -O fair \
        -l "${LOG_LEVEL}" \
        --autoscale="${CELERY_MAX_WORKERS},1"
elif [[ "celery-beat" == "$1" ]]; then
    do_gosu "${APP_USER}:${APP_GROUP}" exec celery \
        -A "${CELERY_APP}" --workdir="${APP_PATH}" beat \
        --loglevel="${LOG_LEVEL}" \
        -s "${CELERY_BEAT_DATABASE_FILE}"
fi

exec "$@"
