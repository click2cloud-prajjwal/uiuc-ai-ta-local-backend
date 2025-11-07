#!/bin/bash

echo "🚀 Starting UIUC Backend Services..."
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Start all services
echo "📦 Starting PostgreSQL, RabbitMQ, and Qdrant..."
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo ""
echo "📊 Service Status:"
echo "===================="

# Check PostgreSQL
if docker exec uiuc-postgres pg_isready -U postgres &> /dev/null; then
    echo "✅ PostgreSQL is running on port 5432"
else
    echo "❌ PostgreSQL failed to start"
fi

# Check RabbitMQ
if docker exec uiuc-rabbitmq rabbitmq-diagnostics ping &> /dev/null; then
    echo "✅ RabbitMQ is running on port 5672"
    echo "   Management UI: http://localhost:15672 (guest/guest)"
else
    echo "❌ RabbitMQ failed to start"
fi

# Check Qdrant
if curl -s http://localhost:6333/ &> /dev/null; then
    echo "✅ Qdrant is running on port 6333"
    echo "   Dashboard: http://localhost:6333/dashboard"
else
    echo "❌ Qdrant failed to start"
fi

echo ""
echo "======================================"
echo "🎉 All services are running!"
echo ""
echo "Connection Details:"
echo "===================="
echo "PostgreSQL:"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  User: postgres"
echo "  Password: postgres"
echo "  Database: uiuc_db"
echo ""
echo "RabbitMQ:"
echo "  Host: localhost"
echo "  Port: 5672"
echo "  User: guest"
echo "  Password: guest"
echo "  Management: http://localhost:15672"
echo ""
echo "Qdrant:"
echo "  Host: localhost"
echo "  Port: 6333"
echo "  Dashboard: http://localhost:6333/dashboard"
echo ""
echo "To stop services: docker-compose down"
echo "To view logs: docker-compose logs -f"