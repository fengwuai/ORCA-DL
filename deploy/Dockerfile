FROM ghcr.io/prefix-dev/pixi:0.46.0 AS build

WORKDIR /app

COPY pixi.toml pixi.lock ./

RUN pixi install -e model && \
    pixi install -e exec && \
    pixi install -e orchestrator

COPY . .

FROM ghcr.io/prefix-dev/pixi:0.46.0 AS runtime

WORKDIR /app

COPY --from=build /app /app

CMD ["pixi", "run", "-e", "orchestrator", "pipeline-serve"]
