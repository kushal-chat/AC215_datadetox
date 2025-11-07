FROM node:20-slim AS builder
WORKDIR /app

# Copy package files from frontend directory
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . ./

# Build the app
RUN npm run build

# Production stage
FROM node:20-slim AS runner
WORKDIR /app

# Copy built assets
COPY --from=builder /app/dist ./dist

# Install serve to run the static site
RUN npm install -g serve

EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]