# ── Build stage ─────────────────────────────────────────────────────────
FROM node:20-alpine AS build
WORKDIR /app

# Install dependencies from lockfile for reproducible builds.
COPY package.json package-lock.json* ./
RUN npm ci

# Build the production bundle.
COPY . .
RUN npm run build

# ── Serve stage ─────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS serve
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -q -O /dev/null http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
