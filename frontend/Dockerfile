# Use Node.js LTS version as base image
FROM node:22-slim

# Set working directory
WORKDIR /src/app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy rest of the application code
COPY . .

# Build Next.js app
RUN npm run build

# Start the application
CMD ["npm", "run", "dev"]
