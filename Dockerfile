#FROM python:3.10-slim AS base # not working with Apple M chip
FROM python:3.10-slim-bullseye AS base 

# Useful packages to handle images, fonts etc.
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends git curl make cmake \
    xz-utils pkg-config build-essential \
    mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev \
    libxi-dev libxrandr-dev libfreetype6-dev libfontconfig1-dev \
    libjpeg-dev libcairo2-dev liblcms2-dev libboost-dev libopenjp2-7-dev \
    libopenjp2-tools

# install poetry
# https://python-poetry.org/docs/configuration/#using-environment-variables
ENV POETRY_HOME="/opt/poetry" \
    # avoid poetry creating virtual environment in the project's root
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV APP_DIR="/application" \
    PATH="$POETRY_HOME/bin:$PATH" \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1
WORKDIR $APP_DIR    

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
# Avoid creating virtual environments in docker
RUN poetry config virtualenvs.create false

RUN poetry install --only main --no-root

COPY ./app ${APP_DIR}/app

# Ensure the predictions directory exists and is writable
RUN mkdir -p ${APP_DIR}/app/webservice_predictions && chmod 777 ${APP_DIR}/app/webservice_predictions

# sensible defaults for the fastapi app
ENV DEBUG=false

EXPOSE 9000

# Run uvicorn programmatically
CMD ["poetry", "run", "python", "-m", "app.main"]
