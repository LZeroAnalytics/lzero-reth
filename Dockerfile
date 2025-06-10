# syntax=docker.io/docker/dockerfile:1.7-labs

FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

LABEL org.opencontainers.image.source=https://github.com/paradigmxyz/reth
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"

# Install system dependencies
RUN apt-get update && apt-get -y upgrade && apt-get install -y libclang-dev pkg-config

FROM chef AS planner
COPY --exclude=.git --exclude=dist . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json

# Build profile, release by default
ARG BUILD_PROFILE=release
ENV BUILD_PROFILE=$BUILD_PROFILE

# Extra Cargo flags
ARG RUSTFLAGS=""
ENV RUSTFLAGS="$RUSTFLAGS"

# Extra Cargo features
ARG FEATURES=""
ENV FEATURES=$FEATURES

# Builds dependencies
RUN cargo chef cook --profile $BUILD_PROFILE --features "$FEATURES" --recipe-path recipe.json

# Build the package for lzero-custom-reth
COPY --exclude=.git --exclude=dist . .
RUN cargo build --profile $BUILD_PROFILE -p lzero-custom-reth

# Copy the built binary to a temporary location
RUN cp /app/target/$BUILD_PROFILE/lzero-custom-reth /app/lzero-custom-reth

# Use Ubuntu as the release image
FROM ubuntu AS runtime
WORKDIR /app

# Copy the built binary
COPY --from=builder /app/lzero-custom-reth /usr/local/bin

# Copy licenses
COPY LICENSE-* ./

# Expose the necessary ports
EXPOSE 30303 30303/udp 9001 8545 8546

# Set the entrypoint to the built binary
ENTRYPOINT ["/usr/local/bin/lzero-custom-reth"]
